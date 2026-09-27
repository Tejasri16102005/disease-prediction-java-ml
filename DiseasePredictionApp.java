package com.teju.ml;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.File;
import java.util.*;

public class DiseasePredictionApp {

    private static final String DATASET = "dataset.csv";

    public static void main(String[] args) throws Exception {
        Instances data = loadDataset(DATASET);

        // Last column ("disease") is the target/class.
        data.setClassIndex(data.numAttributes() - 1);

        RandomForest model = new RandomForest();
        model.setNumIterations(150);
        model.setMaxDepth(8);
        model.setSeed(42);
        model.buildClassifier(data);

        Evaluation evaluation = new Evaluation(data);
        evaluation.crossValidateModel(model, data, 5, new Random(42));

        System.out.printf("5-fold cross-validation accuracy: %.2f%%%n",
                evaluation.pctCorrect());
        System.out.println();

        runPrediction(model, data);
    }

    private static Instances loadDataset(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        return loader.getDataSet();
    }

    private static void runPrediction(RandomForest model, Instances data) throws Exception {
        Scanner scanner = new Scanner(System.in);

        System.out.println("=== Disease Prediction - Java + Machine Learning ===");
        System.out.println("Educational demo only; NOT a medical diagnostic tool.");
        System.out.println();

        List<String> symptoms = new ArrayList<>();
        for (int i = 0; i < data.numAttributes() - 1; i++) {
            symptoms.add(data.attribute(i).name());
            System.out.printf("%2d. %s%n", i + 1,
                    data.attribute(i).name().replace('_', ' '));
        }

        System.out.print("\nEnter symptom numbers separated by commas: ");
        String input = scanner.nextLine();

        Set<Integer> selected = new HashSet<>();
        for (String token : input.split(",")) {
            try {
                int number = Integer.parseInt(token.trim());
                if (number >= 1 && number <= symptoms.size()) {
                    selected.add(number - 1);
                }
            } catch (NumberFormatException ignored) {
            }
        }

        Instance patient = new DenseInstance(data.numAttributes());
        patient.setDataset(data);

        for (int i = 0; i < symptoms.size(); i++) {
            patient.setValue(i, selected.contains(i) ? 1.0 : 0.0);
        }
        patient.setMissing(data.classIndex());

        double[] probabilities = model.distributionForInstance(patient);

        Integer[] indices = new Integer[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            indices[i] = i;
        }

        Arrays.sort(indices, (a, b) ->
                Double.compare(probabilities[b], probabilities[a]));

        System.out.println("\nTop model classifications:");
        for (int i = 0; i < Math.min(3, indices.length); i++) {
            int index = indices[i];
            String label = data.classAttribute().value(index);
            System.out.printf("%d. %s - %.2f%%%n",
                    i + 1, label, probabilities[index] * 100);
        }

        System.out.println("\nNote: This result is for an educational ML demonstration.");
    }
}
