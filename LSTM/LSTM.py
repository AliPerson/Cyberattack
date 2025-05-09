import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import logging
import sys

# Setup logging to file and console
log_filename = "LSTM/lstm_attack_detector.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler(sys.stdout)
                    ])

logging.info("Script started.")

try:
    # File paths
    train_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv"
    test_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv"
    categorical_features = ['proto', 'service', 'state']

    logging.info("Loading training data...")
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Combine to ensure consistent label encoding across both
    combined_data = pd.concat([train_data, test_data], axis=0)

    # Encode categorical features
    encoders = {}
    for col in categorical_features:
        if col in combined_data.columns:
            le = LabelEncoder()
            combined_data[col] = le.fit_transform(combined_data[col])
            encoders[col] = le

    # Save encoders
    joblib.dump(encoders, 'LSTM/lstm_label_encoders.pkl')

    # Separate again after encoding
    train_data_encoded = combined_data.iloc[:len(train_data)]
    test_data_encoded = combined_data.iloc[len(train_data):]

    X_train = train_data_encoded.drop(columns=["id", "label", "attack_cat"])
    y_train = train_data_encoded["label"]

    X_test = test_data_encoded.drop(columns=["id", "label", "attack_cat"])
    y_test = test_data_encoded["label"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    joblib.dump(label_encoder, 'LSTM/lstm_label_encoder.pkl')

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, 'LSTM/lstm_scaler.pkl')

    # Reshape input for LSTM [samples, timesteps, features]
    X_train_lstm = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_lstm = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    logging.info("Building and training LSTM model...")

    model = Sequential([
        LSTM(64, input_shape=(1, X_train.shape[1]), activation='tanh'),
        Dense(64, activation='relu'),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_lstm, y_train_encoded, epochs=10, batch_size=64, validation_split=0.2, verbose=2)

    # Save model in modern format
    model.save('LSTM/lstm_attack_detector.keras')

    logging.info("LSTM model training complete and saved.")

    logging.info("Evaluating on test data...")

    # Evaluate
    y_pred_probs = model.predict(X_test_lstm)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test_encoded, y_pred)
    logging.info(f"Test Accuracy: {acc}")
    logging.info("Classification Report:")
    unique_labels = np.unique(y_test_encoded)
    all_class_names = label_encoder.classes_

    # Convert to strings to avoid type issues
    target_names = [str(all_class_names[i]) for i in unique_labels]

    report = classification_report(y_test_encoded, y_pred, labels=unique_labels, target_names=target_names)
    print("Test Accuracy:", acc)
    print("Classification Report:")
    print(report)

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

logging.info("Script execution completed.")
