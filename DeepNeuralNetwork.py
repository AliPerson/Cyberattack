import sys

# Logger class to redirect stdout to both console and a log file
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")  # "w" overwrites; change to "a" to append
        self.encoding = self.terminal.encoding  # Ensure Keras can access the encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout so that all print output is logged to the specified file.
sys.stdout = Logger("deep_neural_network_attack_detector.log")

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense, Dropout #type: ignore

# Define file paths for training and testing data
train_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv"
test_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv"

# Define the categorical features that need encoding
categorical_features = ['proto', 'service', 'state']

#########################################
#           TRAINING SECTION            #
#########################################

# Load the training data
train_data = pd.read_csv(train_file)

# Separate features and target variable
X_train = train_data.drop(columns=["id", "label", "attack_cat"])
y_train = train_data["label"]

# Encode categorical features in the training set
encoders = {}
for col in categorical_features:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Build the deep neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # For binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training deep neural network...")
# Set verbose=2 to print one line per epoch instead of detailed batch output.
model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=2)
print("Training complete. Model, scaler, and encoders saved.")

# Save the trained model, scaler, and label encoders for later use
model.save('attack_detector_model.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

#########################################
#           TESTING SECTION             #
#########################################

# Load the testing data
test_data = pd.read_csv(test_file)
X_test = test_data.drop(columns=["id", "label", "attack_cat"])
y_test = test_data["label"]

# Load the saved encoders and scaler
encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Apply the same encoding to the test data
for col in categorical_features:
    if col in X_test.columns:
        le = encoders[col]
        # Map unseen categories to -1
        X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale the test data using the saved scaler
X_test_scaled = scaler.transform(X_test)

# Load the saved model
model = tf.keras.models.load_model('attack_detector_model.h5')

# Make predictions and evaluate the model on the test data
y_pred_prob = model.predict(X_test_scaled)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary values

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
