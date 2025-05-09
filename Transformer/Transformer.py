import pandas as pd
import numpy as np
import logging
import sys
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Setup logging
log_filename = "Transformer/transformer_attack_detector.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Script started.")

try:
    # Load data
    logging.info("Loading training data...")
    df = pd.read_csv(r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv")

    # Identify categorical and numerical features
    categorical_features = ['proto', 'service', 'state']
    numerical_features = [col for col in df.columns if col not in categorical_features + ['id', 'label', 'attack_cat']]

    # Encode categorical features
    encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])

    # Split features and target
    X = df[categorical_features + numerical_features]
    y = df['label']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_features])
    X_test_num = scaler.transform(X_test[numerical_features])

    # Prepare categorical features
    X_train_cat = X_train[categorical_features].values
    X_test_cat = X_test[categorical_features].values

    # Define inputs
    inputs = []
    encoded_features = []

    # Categorical inputs
    for i, col in enumerate(categorical_features):
        input_cat = keras.Input(shape=(1,), name=col)
        vocab_size = df[col].nunique()
        embedding_dim = 32
        embedding = layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim)(input_cat)
        embedding = layers.Reshape(target_shape=(embedding_dim,))(embedding)
        inputs.append(input_cat)
        encoded_features.append(embedding)

    # Numerical inputs
    input_num = keras.Input(shape=(len(numerical_features),), name='numerical')
    inputs.append(input_num)
    encoded_features.append(input_num)

    # Concatenate all features
    x = layers.Concatenate()(encoded_features)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    # Define model
    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prepare inputs for model
    train_inputs = {col: X_train_cat[:, i] for i, col in enumerate(categorical_features)}
    train_inputs['numerical'] = X_train_num

    test_inputs = {col: X_test_cat[:, i] for i, col in enumerate(categorical_features)}
    test_inputs['numerical'] = X_test_num

    # Train model
    logging.info("Training Transformer model...")
    model.fit(train_inputs, y_train, epochs=10, batch_size=64, validation_split=0.2)

    # Evaluate model
    logging.info("Evaluating Transformer model...")
    y_pred_prob = model.predict(test_inputs)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Test Accuracy: {acc}")
    logging.info("Classification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    logging.info(report)

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

logging.info("Script execution completed.")