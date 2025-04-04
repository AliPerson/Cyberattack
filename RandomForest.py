import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Define file paths for your training and testing data
train_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\train_file.csv"
test_file = r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv"

# Define the categorical features that need encoding
categorical_features = ['proto', 'service', 'state']

#########################################
#           TRAINING SECTION            #
#########################################

# Load the training data
train_data = pd.read_csv(r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_testing-set.csv")

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

# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Save the trained model, scaler, and label encoders for later use
joblib.dump(clf, 'attack_detector_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoders, 'label_encoders.pkl')

print("Training complete. Model, scaler, and encoders saved.")

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
        X_test[col] = le.transform(X_test[col])

# Scale the test data using the saved scaler
X_test_scaled = scaler.transform(X_test)

# Load the saved model
clf = joblib.load('attack_detector_model.pkl')

# Make predictions and evaluate the model on the test data
y_pred = clf.predict(X_test_scaled)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Neural Network
# SVM
# LSTM
# Transformer
# Graph Neural Network
