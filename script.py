from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import joblib
import os
import numpy as np
import pandas as pd

# Function to load the model from the directory
def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))  # Loading the trained model from the specified directory
    return clf

if __name__ == "__main__":
    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command line arguments to the script
    parser.add_argument("--n_estimators", type = int, default = 100)  # Number of trees in the RandomForest model
    parser.add_argument("--random_state", type = int, default = 0)    # Random seed for reproducibility

    # Data, model, and output directories
    parser.add_argument("--model_dir", type = str, default = os.environ.get("SM_MODEL_DIR"))  # Directory to save the model
    parser.add_argument("--train", type = str, default = os.environ.get("SM_CHANNEL_TRAIN"))  # Directory for training data
    parser.add_argument("--test", type = str, default = os.environ.get("SM_CHANNEL_TEST"))    # Directory for testing data
    parser.add_argument("--train-file", type = str, default = "train-V-1.csv")  # Training data file
    parser.add_argument("--test-file", type = str, default = "test-V-1.csv")    # Testing data file

    args, _ = parser.parse_known_args()

    # Printing the versions of sklearn and joblib for debugging
    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib version: ", joblib.__version__)

    print("[INFO] Reading Data")
    print()

    # Reading the training and testing data into DataFrames
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)  # Getting the feature columns from the training data
    label = features.pop()  # Removing and storing the label column

    print("[INFO] Building training and testing datasets")
    print()

    # Splitting the data into features (X) and target (y)
    X_train = train_df[features]
    y_train = train_df[label]

    X_test = test_df[features]
    y_test = test_df[label]

    print("Column order: ")
    print(features)
    print()

    print("Label column is: ", label)
    print()

    print("Data Shape")
    print()
    print("-------- SHAPE OF THE TRAINING DATA (85%)--------")
    print(X_train.shape)
    print(y_train.shape)

    print("Training the RandomForest Model...")
    print()
    # Training the RandomForest model with the specified hyperparameters
    model = RandomForestClassifier(n_estimators = args.n_estimators, random_state = args.random_state)
    model.fit(X_train, y_train)
    print()

    # Saving the trained model to the specified directory
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at: ", model_path)
    print()

    # Making predictions on the test set
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)  # Calculating accuracy
    test_rep = classification_report(y_test, y_pred_test)  # Generating a classification report

    print()
    print("-------- METRICS RESULT FOR TESTING DATA --------")
    print()
    print("Total rows are: ", X_test.shape[0])
    print("Accuracy Score: ", test_acc)
    print("Test Report: ")
    print(test_rep)