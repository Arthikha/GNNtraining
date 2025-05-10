import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import itertools
import numpy as np
from neo4j import GraphDatabase
import uuid
import json

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Neo4j connection
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.user =os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'testpassword')
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.connected = True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self.connected = False

    def close(self):
        if self.connected:
            self.driver.close()

    def clear_database(self):
        if not self.connected:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_account_node(self, tx, acc_num, features):
        query = (
            "CREATE (a:Account {acc_num: $acc_num, "
            "trans_count: $trans_count, state: $state, zip: $zip, city_pop: $city_pop, "
            "unix_time_mean: $unix_time_mean, time_span: $time_span, ip_trans_freq: $ip_trans_freq, "
            "ip_address: $ip_address})"
        )
        # Ensure trans_count is an integer and non-negative
        # features['trans_count'] = int(max(features['trans_count'], 0))
        # Ensure ip_address is a string
        features['ip_address'] = str(features['ip_address'])
        tx.run(query, **features)

    def create_edge(self, tx, src_idx, dst_idx, edge_type):
        query = (
            "MATCH (a:Account), (b:Account) "
            "WHERE a.acc_num = $src_acc AND b.acc_num = $dst_acc "
            f"CREATE (a)-[:{edge_type}]->(b)"
        )
        tx.run(query, src_acc=idx_to_acc[src_idx], dst_acc=idx_to_acc[dst_idx])

    def create_fraud_ring(self, tx, ring_id, accounts):
        query = (
            "CREATE (r:FraudRing {ring_id: $ring_id}) "
            "WITH r "
            "UNWIND $accounts AS acc "
            "MATCH (a:Account {acc_num: acc.acc_num}) "
            "CREATE (a)-[:IN_FRAUD_RING]->(r)"
        )
        tx.run(query, ring_id=ring_id, accounts=accounts)

# Initialize Neo4j
neo4j_conn = Neo4jConnection()

# Load dataset
dataset_path = os.path.join(os.path.dirname(__file__), '../../dataset/fraud_with_rings_2.csv')
data_df = pd.read_csv(dataset_path)

# Map accounts to node indices
unique_accounts = data_df["acc_num"].unique()
acc_to_idx = {acc: idx for idx, acc in enumerate(unique_accounts)}
idx_to_acc = {idx: acc for idx, acc in enumerate(unique_accounts)}
data_df["node_index"] = data_df["acc_num"].map(acc_to_idx)

# Feature engineering
ip_trans_freq = data_df.groupby(["acc_num", "ip_address"])["trans_num"].count().reset_index()
ip_trans_freq = ip_trans_freq.groupby("acc_num")["trans_num"].sum().reset_index()
ip_trans_freq.columns = ["acc_num", "ip_trans_freq"]

agg_features = data_df.groupby("acc_num").agg({
    "trans_num": "count",
    "state": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
    "zip": "first",
    "city_pop": "mean",
    "unix_time": ["mean", lambda x: x.max() - x.min()],
    "ip_address": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
}).reset_index()

# Flatten column names
agg_features.columns = [
    "acc_num", "trans_count", "state", "zip",
    "city_pop", "unix_time_mean", "time_span", "ip_address"
]

# Merge ip_trans_freq
agg_features = agg_features.merge(ip_trans_freq, on="acc_num", how="left")
agg_features["ip_trans_freq"] = agg_features["ip_trans_freq"].fillna(0)

# Create a column for encoded ip_address for model input
agg_features["ip_address_encoded"] = agg_features["ip_address"].astype("category").cat.codes

# Encode state column
agg_features["state"] = agg_features["state"].astype("category").cat.codes

# Normalize numerical features (exclude trans_count to keep it integer)
scaler = StandardScaler()
numerical_cols = ["city_pop", "unix_time_mean", "time_span", "ip_trans_freq"]
agg_features[numerical_cols] = scaler.fit_transform(agg_features[numerical_cols])

# Ensure trans_count is integer and non-negative
agg_features["trans_count"] = agg_features["trans_count"].astype(int)
# agg_features["trans_count"] = agg_features["trans_count"].clip(lower=0)

# Node feature matrix (use encoded ip_address)
x = torch.tensor(
    agg_features[["trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_trans_freq", "ip_address_encoded"]].values,
    dtype=torch.float
)

# Build edges
edges = set()
# zip_groups = data_df.groupby("zip")["node_index"].apply(list)
# ip_groups = data_df.groupby("ip_address")["node_index"].apply(list)

# def add_ip_edges():
#     for ip, nodes in ip_groups.items():
#         if len(nodes) > 1:
#             trans_counts = data_df[data_df["ip_address"] == ip].groupby("node_index")["trans_num"].count()
#             nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
#             for src, dst in itertools.combinations(nodes, 2):
#                 edges.add((src, dst))
#                 edges.add((dst, src))

# def add_zip_edges():
#     for zip_code, nodes in zip_groups.items():
#         if len(nodes) > 1:
#             trans_counts = data_df[data_df["zip"] == zip_code].groupby("node_index")["trans_num"].count()
#             nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
#             for src, dst in itertools.combinations(nodes, 2):
#                 edges.add((src, dst))
#                 edges.add((dst, src))

def add_zip_and_ip_edges():
        # Add IP-based edges
        for ip, nodes in ip_groups.items():
            if len(nodes) > 1:
                trans_counts = data_df[data_df["ip_address"] == ip].groupby("node_index")["trans_num"].count()
                nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
                for src, dst in itertools.combinations(nodes, 2):
                    edges.add((src, dst))
                    edges.add((dst, src))

        # Add ZIP-based edges with stricter condition
        for zip_code, nodes in zip_groups.items():
            if len(nodes) > 1:
                trans_counts = data_df[data_df["zip"] == zip_code].groupby("node_index")["trans_num"].count()
                nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
                # Only add ZIP edges if nodes also share an IP or state
                for src, dst in itertools.combinations(nodes, 2):
                    src_acc = idx_to_acc[src]
                    dst_acc = idx_to_acc[dst]
                    src_ip = data_df[data_df["acc_num"] == src_acc]["ip_address"].iloc[0]
                    dst_ip = data_df[data_df["acc_num"] == dst_acc]["ip_address"].iloc[0]
                    src_state = data_df[data_df["acc_num"] == src_acc]["state"].iloc[0]
                    dst_state = data_df[data_df["acc_num"] == dst_acc]["state"].iloc[0]
                    if src_ip == dst_ip or src_state == dst_state:
                        edges.add((src, dst))
                        edges.add((dst, src))

def add_transaction_edges():
    data_df_sorted = data_df.sort_values("unix_time")
    for ip, group in data_df_sorted.groupby("ip_address"):
        times = group["unix_time"].values
        nodes = group["node_index"].values
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] < 900:  # 15 minutes
                src, dst = nodes[i], nodes[i + 1]
                edges.add((src, dst))
                edges.add((dst, src))

# add_ip_edges()
# add_zip_edges()
add_zip_and_ip_edges()
add_transaction_edges()

edge_index = torch.tensor(list(edges), dtype=torch.long).T

# Encode labels
account_labels = data_df.groupby("acc_num")["fraud_ring_id"].first().reset_index()
account_labels["fraud_ring_id"] = account_labels["fraud_ring_id"].fillna(0)
label_encoder = LabelEncoder()
account_labels["fraud_ring_id"] = label_encoder.fit_transform(account_labels["fraud_ring_id"])
y = torch.tensor(account_labels["fraud_ring_id"].values, dtype=torch.long)

# Compute class weights
class_counts = np.bincount(y.numpy())
class_weights = 1.0 / class_counts
class_weights = np.clip(class_weights / class_weights.sum() * len(class_counts), 0.1, 5.0)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Create graph data object
data = Data(x=x, edge_index=edge_index, y=y)

# Train-test split
train_indices, test_indices = train_test_split(range(data.num_nodes), test_size=0.2, random_state=42)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_indices] = True
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[test_indices] = True

# Define model
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGE(data.num_features, hidden_channels=128, out_channels=y.max().item() + 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Import graph to Neo4j
def import_graph_to_neo4j():
    if not neo4j_conn.connected:
        return
    neo4j_conn.clear_database()
    with neo4j_conn.driver.session() as session:
        for _, row in agg_features.iterrows():
            features = row.to_dict()
            # Use original ip_address for Neo4j, not the encoded version
            features['ip_address'] = str(features['ip_address'])
            session.execute_write(lambda tx: neo4j_conn.create_account_node(tx, features['acc_num'], features))
        for src, dst in edges:
            session.execute_write(lambda tx: neo4j_conn.create_edge(tx, src, dst, "RELATED"))

import_graph_to_neo4j()

# Training
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop with early stopping
best_loss = float('inf')
patience = 30
epochs_without_improvement = 0
model_path = "best_model.pt"

for epoch in range(1, 201):
    loss = train()
    if loss < best_loss:
        best_loss = loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), model_path)
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        break

# Load best model
try:
    model.load_state_dict(torch.load(model_path))
except FileNotFoundError:
    print("No saved model found, using last trained state")

# Predict and extract fraud rings
def get_fraud_rings():
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()

    fraud_rings = defaultdict(list)
    for idx, pred_label in enumerate(pred):
        if pred_label != 0:  # Exclude non-fraud
            fraud_rings[pred_label].append(idx_to_acc[idx])

    # Output format: list of acc_num
    output_rings = [
        {
            "fraud_ring_id": int(label_encoder.inverse_transform([ring_id])[0]),
            "accounts": sorted(acc_list)
        }
        for ring_id, acc_list in sorted(fraud_rings.items(), key=lambda x: x[0])
    ]

    # Neo4j format: list of {acc_num: value}
    neo4j_rings = [
        {
            "fraud_ring_id": int(label_encoder.inverse_transform([ring_id])[0]),
            "accounts": [{"acc_num": acc_num} for acc_num in sorted(acc_list)]
        }
        for ring_id, acc_list in sorted(fraud_rings.items(), key=lambda x: x[0])
    ]

    return output_rings, neo4j_rings

# Save fraud rings to Neo4j
def save_fraud_rings_to_neo4j(fraud_rings):
    if not neo4j_conn.connected:
        return
    with neo4j_conn.driver.session() as session:
        for ring in fraud_rings:
            session.execute_write(lambda tx: neo4j_conn.create_fraud_ring(tx, ring["fraud_ring_id"], ring["accounts"]))

# Output and save results
output_rings, neo4j_rings = get_fraud_rings()
print("\nPredicted Fraud Rings:")
for ring in output_rings:
    print(ring)

save_fraud_rings_to_neo4j(neo4j_rings)

# Save to JSON
with open("fraud_rings.json", "w") as f:
    json.dump(output_rings, f, indent=2)

print("\nRun this Cypher query in Neo4j to visualize fraud rings:")
print("MATCH (a:Account)-[r:IN_FRAUD_RING]->(f:FraudRing) RETURN a, r, f LIMIT 50")
print("Neo4j credentials: user=neo4j, password=testpassword, URI=bolt://neo4j:7687")

# Close Neo4j connection
neo4j_conn.close()