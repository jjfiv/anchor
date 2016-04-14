package cc.mallet.anchor;

import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.util.Randoms;
import ciir.jfoley.chai.collections.list.IntList;
import ciir.jfoley.chai.io.LinesIterable;
import ciir.jfoley.chai.string.StrUtil;
import gnu.trove.TIntIntHashMap;
import org.lemurproject.galago.core.parse.TagTokenizer;
import org.lemurproject.galago.core.util.WordLists;
import org.lemurproject.galago.utility.Parameters;

import java.io.IOException;
import java.util.*;

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
        FixedSparseRandomProjection fmatrix = new FixedSparseRandomProjection(alphabet, numProjections, nonZeros, random);
        loadFSRP(fmatrix, instances);
        matrix = fmatrix;
      }
    }

    //matrix.load(instances);
    //System.out.println("built matrix");
    matrix.rowNormalize();
    //System.out.println("normalized");

    return matrix;
  }

  public static void loadFSRP(FixedSparseRandomProjection matrix, List<IntList> instances) {
    for (IntList instance: instances) {
      //HashMap<Integer,Integer> typeCounts = new HashMap<Integer,Integer>();

      TIntIntHashMap typeCounts = new TIntIntHashMap();
      int length = instance.size();

      // Skip short documents
      //if (length < 10) { continue; }

      matrix.totalTokens += length;

      for (int position = 0; position < length; position++) {
        int type = instance.getQuick(position);
        matrix.wordCounts[type]++;
        typeCounts.adjustOrPutValue(type, 1, 1);
      }

      double inverseNumPairs = 2.0 / (length * (length - 1));
      double coefficient = matrix.squareRootSparsity * 2.0 / (length * (length - 1));

      int[] types = typeCounts.keys();
      int[] counts = typeCounts.getValues();

      // Multiply the counts by the random projection matrix
      double[] projectedSignature = new double[matrix.numColumns];
      for (int i = 0; i < types.length; i++) {
        int type = types[i];
        matrix.documentFrequencies[type]++;

        for (int j = 0; j < matrix.nonZeros; j++) {
          int projectionIndex = matrix.projectionMatrix[type][j];
          if (projectionIndex >= matrix.numColumns) {
            // If it's in this range, it should be negated
            projectedSignature[projectionIndex - matrix.numColumns] += -counts[i];
          }
          else {
            projectedSignature[projectionIndex] += counts[i];
          }
        }
        matrix.rowSums[type] += inverseNumPairs * counts[i] * (length - counts[i]);
      }

      // Rescale for the sparse random projection and the length of the document
      for (int j = 0; j < matrix.numColumns; j++) {
        projectedSignature[j] *= coefficient;
      }

      // Now multiply by the counts again, subtracting the term for the word itself
      for (int i = 0; i < types.length; i++) {
        int type = types[i];
        for (int denseIndex = 0; denseIndex < matrix.numColumns; denseIndex++) {
          matrix.weights[type][denseIndex] += counts[i] * projectedSignature[denseIndex];
        }
        // Remove the diagonal elements as needed
        for (int sparseIndex = 0; sparseIndex < matrix.nonZeros; sparseIndex++) {
          int denseIndex = matrix.projectionMatrix[type][sparseIndex];
          if (denseIndex >= matrix.numColumns) {
            matrix.weights[type][denseIndex - matrix.numColumns] -= -coefficient * counts[i] * counts[i];
          }
          else {
            matrix.weights[type][denseIndex] -= coefficient * counts[i] * counts[i];
          }
        }
      }

    }
    //System.err.format("%d ms\n", (System.currentTimeMillis() - startTime));
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

    int randomProjections = argp.get("randomProjections", 1000);
    double randomProjectionSparsity = argp.get("randomProjectionDensity", 0.01);
    Randoms random = new Randoms(argp.get("seed", 13));
    final int depth = argp.get("depth", 20000);
    final int numTopics = argp.get("numTopics", 200);
    final int minimumDocumentFrequency = argp.get("minimumDocumentFrequency", 10);
    final int minWordFrequency = argp.get("minWordFrequency", 10);
    final int maxWordFrequency = argp.get("maxWordFrequency", 100);
    final int maxWordIterations = argp.get("maxWordIterations", 50);
    final double minimumWordChange = argp.get("minWordChange", 0.001);

    TagTokenizer tok = new TagTokenizer();
    Set<String> stopwords = WordLists.getWordListOrDie("inquery");

    Alphabet alphabet = new Alphabet(1000);
    List<IntList> texts = new ArrayList<>();

    HashMap<String, String> tf10Qs = new HashMap<>();
    try (LinesIterable lines = LinesIterable.fromFile("/home/jfoley/code/controversyNews/signal_tf10.txt.gz")) {
      for (String line : lines) {
        String[] col = line.split("\t");
        if(col.length != 2) continue;
        tf10Qs.put(col[0], col[1]);
      }
    }

    TIntIntHashMap freqs = new TIntIntHashMap();
    //try (LinesIterable docs = LinesIterable.fromFile(argp.get("input", "/home/jfoley/code/controversyNews/scripts/clueweb/clue09b.seed.jsonl.gz"))) {
      for (String doc : tf10Qs.values()) {
        //Parameters jdoc = Parameters.parseString(doc);
        //String raw_html = jdoc.getString("content");
        //String text = StrUtil.collapseSpecialMarks(Jsoup.parse(raw_html).text());
        String text = doc;
        List<String> tdoc = tok.tokenize(text).terms;
        IntList ids = new IntList(tdoc.size());
        for (String term : tdoc) {
          if(stopwords.contains(term)) continue;
          if(StrUtil.looksLikeInt(term)) continue;
          if("com".equals(term)) continue;
          int id = alphabet.lookupIndex(term, true);
          freqs.adjustOrPutValue(id, 1, 1);
          ids.push(id);
        }
        texts.add(ids);

        if(texts.size() > depth) break;
      }
    //}
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
    recover.wordIterations = maxWordIterations;
    recover.wordConvergenceZero = minimumWordChange;

    end = System.nanoTime();
    System.out.println("Total Time: "+((end - start) / 1e9)+"s");

    IntList sparseWords = new IntList();
    for (int i = 0; i < matrix.numWords; i++) {
      if(freqs.get(i) < minWordFrequency) continue;
      if(freqs.get(i) > maxWordFrequency) continue;
      sparseWords.push(i);
    }
    IDSorter[][] topicSortedWords = new IDSorter[recover.numTopics][sparseWords.size()];
    //PrintStream out = System.out;
    //try (PrintWriter out = IO.openPrintWriter("matrix.out.gz")) {

      for (int wi = 0; wi < sparseWords.size(); wi++) {
        int word = sparseWords.getQuick(wi);
        if(freqs.get(word) < minWordFrequency) continue;
        double wordProb = matrix.unigramProbability(word);
        double[] weights = recover.recover(word);

        for (int topic = 0; topic < weights.length; topic++) {
          //out.format("%d:%.8f ", topic, wordProb * weights[topic]);
          //out.print(wordProb * weights[topic] + "\t");
          topicSortedWords[topic][wi] = new IDSorter(word, wordProb * weights[topic]);
        }
        //out.println();

        if (wi > 0 && wi % 1000 == 0) {
          System.err.format("[%d] %s\n", word, alphabet.lookupObject(word));
        }
      }
    //}

    end = System.nanoTime();
    System.out.println("Total Topic Time: "+((end - start) / 1e9)+"s");

    for (int topic = 0; topic < recover.numTopics; topic++) {
      Arrays.sort(topicSortedWords[topic], (lhs, rhs) -> {
        int cmp = Double.compare(lhs.getWeight(), rhs.getWeight());
        if(cmp == 0)
          return Integer.compare(lhs.getID(), rhs.getID());
        return -cmp;
      });
      StringBuilder builder = new StringBuilder();
      for (int i = 0; i < 10; i++) {
        builder.append(alphabet.lookupObject(topicSortedWords[topic][i].getID()) + " ");
      }
      System.out.format("%d\t%s\t%s\n", topic, alphabet.lookupObject(orthogonalizer.basisVectorIndices[topic]), builder);
    }

    end = System.nanoTime();
    System.out.println("Total Topic Time: "+((end - start) / 1e9)+"s");

  }
}
