import sys

# --------------------------
# Custom Logger for Logging
# --------------------------
# This Logger class redirects all print output to a log file and console
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.encoding = self.terminal.encoding  # Keras and sklearn may check encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect all output to log file
sys.stdout = Logger("SVM/svm_attack_detector.log")

# --------------------------
# Import Required Libraries
# --------------------------
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ----------------------------------------
# Define File Paths for Dataset (Training & Testing)
# ----------------------------------------
train_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv"
test_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv"

# -----------------------------
# Categorical Features to Encode
# -----------------------------
categorical_features = ['proto', 'service', 'state']

# --------------------------
#        TRAINING
# --------------------------

print("Loading training data...")
# Load the training CSV file into a pandas DataFrame
train_data = pd.read_csv(train_file)

# Drop unnecessary columns and separate features (X) from the label (y)
X_train = train_data.drop(columns=["id", "label", "attack_cat"])
y_train = train_data["label"]

# Encode each categorical column using LabelEncoder
encoders = {}
for col in categorical_features:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])  # Convert categories to numbers
        encoders[col] = le  # Save encoder for future use

# Scale the numeric features to standardize data (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# -------------------------------
# Train the Support Vector Machine
# -------------------------------
print("Training SVM classifier...")
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')  # You can tune C and gamma
svm_model.fit(X_train_scaled, y_train)  # Train the model
print("Training complete.")

# Save the trained model and pre-processing tools
joblib.dump(svm_model, 'SVM/svm_attack_detector_model.pkl')
joblib.dump(scaler, 'SVM/svm_scaler.pkl')
joblib.dump(encoders, 'SVM/svm_label_encoders.pkl')
print("Model, scaler, and encoders saved.")

# --------------------------
#        TESTING
# --------------------------

print("Loading testing data...")
# Load the testing data
test_data = pd.read_csv(test_file)
X_test = test_data.drop(columns=["id", "label", "attack_cat"])
y_test = test_data["label"]

# Reload the saved encoders and scaler
encoders = joblib.load('SVM/svm_label_encoders.pkl')
scaler = joblib.load('SVM/svm_scaler.pkl')

# Encode categorical columns using the saved encoders
for col in categorical_features:
    if col in X_test.columns:
        le = encoders[col]
        # Handle unseen categories by assigning them a placeholder (-1)
        X_test[col] = X_test[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale the test data using the saved scaler
X_test_scaled = scaler.transform(X_test)

# Load the trained SVM model
svm_model = joblib.load('SVM/svm_attack_detector_model.pkl')

# Make predictions using the SVM model
y_pred = svm_model.predict(X_test_scaled)

# Evaluate the model
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))