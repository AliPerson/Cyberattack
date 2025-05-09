import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import classification_report, accuracy_score
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('GNN/graph_neural_network_attack_detector.log'), logging.StreamHandler()]
)

logging.info("Script started.")

try:
    # Load data
    logging.info("Loading training data...")
    df = pd.read_csv(r"C:\Users\sabs5\OneDrive - SUNY Brockport\Classes\Independent Study\UNSWNB15\UNSW_NB15_training-set.csv")

    # Preprocess
    categorical_features = ['proto', 'service', 'state']
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["id", "label", "attack_cat"])
    y = df["label"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create graph using k-NN
    logging.info("Building graph structure...")
    k = 5
    knn_graph = kneighbors_graph(X_scaled, k, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.vstack(knn_graph.nonzero()), dtype=torch.long)

    # Create PyG Data object
    data = Data(
        x=torch.tensor(X_scaled, dtype=torch.float),
        y=torch.tensor(y.values, dtype=torch.long),
        edge_index=edge_index
    )

    # Define GNN model
    class GCN(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            return x

    # Train/test split
    logging.info("Splitting dataset...")
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    split = int(0.8 * num_nodes)
    train_idx = torch.tensor(indices[:split], dtype=torch.long)
    test_idx = torch.tensor(indices[split:], dtype=torch.long)

    # Model
    model = GCN(input_dim=X.shape[1], hidden_dim=64, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    logging.info("Training GNN model...")
    model.train()
    for epoch in range(1, 51):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()
        logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluation
    logging.info("Evaluating GNN model...")
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out[test_idx].argmax(dim=1)
        true = data.y[test_idx]

        acc = accuracy_score(true, pred)
        logging.info(f"Test Accuracy: {acc}")
        logging.info("Classification Report:")
        report = classification_report(true, pred)
        print(report)
        logging.info(report)

except Exception as e:
    logging.error(f"Error occurred: {str(e)}")

logging.info("Script execution completed.")