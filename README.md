# Disease Prediction using Java + Machine Learning

A Java machine-learning project that classifies a possible disease label from selected symptoms.

> **Important:** This is an educational ML project. The included dataset is synthetic and the output must not be used for medical diagnosis or treatment.

## Tech Stack

- Java 17
- Maven
- Weka 3.8.6
- Random Forest
- CSV dataset

## Architecture

```text
dataset.csv
    |
    v
CSVLoader (Weka)
    |
    v
Training Data
    |
    v
Random Forest Classifier
    |
    +----> 5-fold Cross Validation
    |
    v
User Symptom Input
    |
    v
Feature Vector
    |
    v
Model Prediction
    |
    v
Top-3 Classifications
```

## ML Concepts Demonstrated

1. Feature representation using binary symptom columns.
2. Target/class variable: `disease`.
3. Random Forest classification.
4. 5-fold cross-validation.
5. Accuracy evaluation.
6. Probability distribution for top predictions.
7. Java integration with a machine-learning library.

## Project Structure

```text
disease-prediction-java-ml/
├── dataset.csv
├── pom.xml
├── README.md
└── src/
    └── main/
        └── java/
            └── com/
                └── teju/
                    └── ml/
                        └── DiseasePredictionApp.java
```

## Requirements

- JDK 17+
- Maven 3.8+

Check:

```bash
java -version
mvn -version
```

## Run

From the project root:

```bash
mvn clean compile
mvn exec:java
```

The program will display the symptoms. Enter numbers such as:

```text
1,2,3,7
```

The Random Forest model then displays its top classifications.

## Why Weka?

Weka is a Java-based machine-learning toolkit. It lets the project demonstrate real ML while keeping the application completely Java-based.

## Resume Version

**Disease Prediction using Machine Learning** | Java, Weka, Random Forest, Maven

- Developed a Java-based machine-learning application that classifies disease labels from symptom features.
- Represented symptoms as binary features and trained a Random Forest classification model using Weka.
- Evaluated the model using 5-fold cross-validation and classification accuracy.
- Implemented interactive symptom input and probability-based top-3 predictions.

## Interview Explanation

"I built this project in Java using the Weka machine-learning library. The dataset contains symptoms represented as binary features and a disease label as the target. I trained a Random Forest classifier and used 5-fold cross-validation to evaluate it. During prediction, the user selects symptoms, Java converts them into a feature vector, and the trained model returns the top classifications with their probability distribution."

## Future Improvements

- Use a properly sourced public dataset instead of synthetic data.
- Add a Java Swing or JavaFX GUI.
- Compare Random Forest with Decision Tree, SVM, and Logistic Regression.
- Add confusion matrix and precision/recall/F1 metrics.
- Add database storage for prediction history.
- Build a REST API with Spring Boot.

## License

For educational use.
