package cc.mallet.anchor;

import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.util.Randoms;
import ciir.jfoley.chai.collections.list.IntList;
import ciir.jfoley.chai.io.IO;
import ciir.jfoley.chai.io.LinesIterable;
import ciir.jfoley.chai.string.StrUtil;
import gnu.trove.TIntIntHashMap;
import org.jsoup.Jsoup;
import org.lemurproject.galago.core.parse.TagTokenizer;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * @author jfoley
 */
public class SpectralJSONL {

  public static BigramProbabilityMatrix getMatrix(List<IntList> instances, Alphabet alphabet, int numProjections, double sparsity, Randoms random) {

    BigramProbabilityMatrix matrix;

    if (numProjections == 0) {
      matrix = new BigramProbabilityMatrix(alphabet);
      load(matrix, instances);
    }
    else {
      if (sparsity == 1.0) {
        matrix = new GaussianRandomProjection(alphabet, numProjections, random);
      }
      else {
        // For this projection, sparsity is the number of non-zeros
        int nonZeros = (int) Math.floor(sparsity * numProjections);
        matrix = new FixedSparseRandomProjection(alphabet, numProjections, nonZeros, random);
      }
    }

    //matrix.load(instances);
    //System.out.println("built matrix");
    matrix.rowNormalize();
    //System.out.println("normalized");

    return matrix;
  }

  public static void load(BigramProbabilityMatrix matrix, List<IntList> instances) {
    TIntIntHashMap typeCounts = new TIntIntHashMap();
    for (IntList text : instances) {
      int length = text.size();
      if(length < 10) continue;
      matrix.totalTokens += length;
      for (int type : text) {
        matrix.wordCounts[type]++;
        typeCounts.adjustOrPutValue(type, 1, 1);
      }

      double coefficient = 2.0 / (length * (length -1));

      int[] types = typeCounts.keys();
      int[] counts = typeCounts.getValues();

      for (int i = 0; i < types.length; i++) {
        int firstType = types[i];
        matrix.documentFrequencies[firstType]++;

        for (int j = 0; j < types.length; j++) {
          int secondType = types[j];
          double value = coefficient * counts[i] * counts[j];

          if(value < 0.0) {
            String err = String.format("(2.0 / (%d * (%d - 1))) * %d * %d\n", length, length, counts[i], counts[j]);
            throw new RuntimeException(err);
          }

          matrix.weights[firstType][secondType] += value;
          matrix.weights[secondType][firstType] += value;
        }
      }
    }
  }



  public static void main(String[] args) throws IOException {
    Parameters argp = Parameters.parseArgs(args);

    int randomProjections = argp.get("randomProjections", 0);
    double randomProjectionSparsity = argp.get("randomProjectionDensity", 1.0);
    Randoms random = new Randoms(argp.get("seed", 13));
    final int depth = argp.get("depth", 20);
    final int numTopics = argp.get("numTopics", 10);
    final int minimumDocumentFrequency = argp.get("minimumDocumentFrequency", 1);

    TagTokenizer tok = new TagTokenizer();
    Set<String> stopwords = WordLists.getWordListOrDie("inquery");

    Alphabet alphabet = new Alphabet(1000);
    List<IntList> texts = new ArrayList<>();

    try (LinesIterable docs = LinesIterable.fromFile(argp.get("input", "/home/jfoley/code/controversyNews/scripts/clueweb/clue09b.seed.jsonl.gz"))) {
      for (String doc : docs) {
        Parameters jdoc = Parameters.parseString(doc);
        String raw_html = jdoc.getString("content");
        String text = StrUtil.collapseSpecialMarks(Jsoup.parse(raw_html).text());
        List<String> tdoc = tok.tokenize(text).terms;
        IntList ids = new IntList(tdoc.size());
        for (String term : tdoc) {
          if(stopwords.contains(term)) continue;
          if(StrUtil.looksLikeInt(term)) continue;
          if("com".equals(term)) continue;

          ids.push(alphabet.lookupIndex(term, true));
        }
        texts.add(ids);

        if(docs.getLineNumber() > depth) break;
      }
    }
    System.out.println("Load & Tokenize, K="+alphabet.size());

    long start = System.nanoTime();
    long end;
    BigramProbabilityMatrix matrix = getMatrix(texts, alphabet, randomProjections, randomProjectionSparsity, random);

    end = System.nanoTime();
    System.out.println("Load Matrix Cumulative Time: "+((end - start) / 1e9)+"s");

    StabilizedGS orthogonalizer = new StabilizedGS(matrix, minimumDocumentFrequency);
    System.out.format("%d / %d words above document cutoff for anchor words\n", orthogonalizer.getNumInterestingRows(), matrix.numWords);
    System.out.println("Finding anchor words");

    end = System.nanoTime();
    System.out.println("Load Orthogonalizer Cumulative Time: "+((end - start) / 1e9)+"s");
    orthogonalizer.orthogonalize(numTopics);

    end = System.nanoTime();
    System.out.println("Orthogonalizer.orthogonalize Cumulative Time: "+((end - start) / 1e9)+"s");


    // The orthogonalization modified the matrix in place, so reload it.
    orthogonalizer.clearMatrix();  // save some memory
    matrix = getMatrix(texts, alphabet, randomProjections, randomProjectionSparsity, random);
    SpectralLDA recover = new SpectralLDA(matrix, orthogonalizer);

    end = System.nanoTime();
    System.out.println("Total Time: "+((end - start) / 1e9)+"s");

    IDSorter[][] topicSortedWords = new IDSorter[recover.numTopics][matrix.numWords];
    //PrintStream out = System.out;
    try (PrintWriter out = IO.openPrintWriter("matrix.out.gz")) {

      for (int word = 0; word < matrix.numWords; word++) {
        double wordProb = matrix.unigramProbability(word);
        double[] weights = recover.recover(word);

        for (int topic = 0; topic < weights.length; topic++) {
          //out.format("%d:%.8f ", topic, wordProb * weights[topic]);
          out.print(wordProb * weights[topic] + "\t");
          topicSortedWords[topic][word] = new IDSorter(word, wordProb * weights[topic]);
        }
        out.println();

        if (word > 0 && word % 1000 == 0) {
          System.out.format("[%d] %s\n", word, alphabet.lookupObject(word));
        }
      }
    }

    end = System.nanoTime();
    System.out.println("Total Topic Time: "+((end - start) / 1e9)+"s");

    for (int topic = 0; topic < recover.numTopics; topic++) {
      Arrays.sort(topicSortedWords[topic]);
      StringBuilder builder = new StringBuilder();
      for (int i = 0; i < 20; i++) {
        builder.append(alphabet.lookupObject(topicSortedWords[topic][i].getID()) + " ");
      }
      System.out.format("%d\t%s\t%s\n", topic, alphabet.lookupObject(orthogonalizer.basisVectorIndices[topic]), builder);
    }

    end = System.nanoTime();
    System.out.println("Total Topic Time: "+((end - start) / 1e9)+"s");

  }
}
