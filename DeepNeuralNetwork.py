import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np
import logging
import sys

# Configure logging to log both to file and console
log_file = "attack_detector_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Start logging
logging.info("Starting Attack Detector Script")

try:
    # Verify TensorFlow installation
    logging.info(f"Using TensorFlow version: {tf.__version__}")

    # Define file paths for training and testing data
    train_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv"
    test_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv"

    # Define categorical features that need encoding
    categorical_features = ['proto', 'service', 'state']

    logging.info("Loading training data...")
    train_data = pd.read_csv(train_file)

    # Separate features and target variable
    X = train_data.drop(columns=["id", "label", "attack_cat"])
    y = train_data["label"]

    # Encode categorical features
    encoders = {}
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    # Normalize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    logging.info("Building neural network model...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),  # Dropout to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification output
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    logging.info("Training the model...")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

    # Save the trained model, scaler, and encoders
    model.save('attack_detector_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoders, 'label_encoders.pkl')

    logging.info("Training complete. Model, scaler, and encoders saved.")

    #########################################
    #           TESTING SECTION             #
    #########################################

    logging.info("Loading testing data...")
    test_data = pd.read_csv(test_file)

    # Separate features and target variable
    X_test = test_data.drop(columns=["id", "label", "attack_cat"])
    y_test = test_data["label"]

    # Load the saved encoders and scaler
    encoders = joblib.load('label_encoders.pkl')
    scaler = joblib.load('scaler.pkl')

    # Apply the same encoding to the test data
    for col in categorical_features:
        if col in X_test.columns:
            le = encoders[col]
            X_test[col] = X_test[col].map(lambda s: le.transform([s])[0] if s in le.classes_ else -1)

    # Scale the test data using the saved scaler
    X_test_scaled = scaler.transform(X_test)

    # Load the trained model
    model = keras.models.load_model('attack_detector_model.h5')

    # Make predictions
    logging.info("Making predictions on test data...")
    y_pred_prob = model.predict(X_test_scaled)
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary values

    # Output results
    test_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info("Classification Report:\n" + class_report)

except Exception as e:
    logging.error(f"An error occurred: {str(e)}", exc_info=True)