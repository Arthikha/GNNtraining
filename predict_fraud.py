import os
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from collections import defaultdict
import itertools
from neo4j import GraphDatabase


# Neo4j connection class
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://neo4j:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'testpassword')
        self.connected = False
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.connected = True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self.connected = False


    def close(self):
        if self.connected:
            self.driver.close()
            print("Neo4j connection closed.")

    def create_account_node(self, tx, acc_num, features):
        query = (
            "CREATE (a:Account {acc_num: $acc_num, amt_mean: $amt_mean, amt_std: $amt_std, "
            "trans_count: $trans_count, state: $state, zip: $zip, city_pop: $city_pop, "
            "unix_time_mean: $unix_time_mean, time_span: $time_span, ip_trans_freq: $ip_trans_freq, "
            "ip_address: $ip_address})"
        )

        features['trans_count'] = int(max(features['trans_count'], 0))
        features['ip_address'] = str(features['ip_address'])
        features['acc_num'] = str(features['acc_num'])


    def create_edge(self, tx, src_acc, dst_acc, edge_type):
        query = (
            "MATCH (a:Account {acc_num: $src_acc}), (b:Account {acc_num: $dst_acc}) "
            f"CREATE (a)-[:{edge_type}]->(b)"
        )
        tx.run(query, src_acc=str(src_acc), dst_acc=str(dst_acc))


    def create_fraud_ring(self, tx, ring_id, accounts):
        query = (
            "CREATE (r:FraudRing {ring_id: $ring_id}) "
            "WITH r "
            "UNWIND $accounts AS acc "
            "MATCH (a:Account {acc_num: acc.acc_num}) "
            "CREATE (a)-[:IN_FRAUD_RING]->(r)"
        )
        tx.run(query, ring_id=ring_id, accounts=accounts)


# Define the GraphSAGE model
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
    

# Function to process transactions and predict fraud rings
def process_and_predict(transactions):
    # Convert transactions to DataFrame
    data_df = pd.DataFrame(transactions)

    # Cast acc_num to string
    data_df['acc_num'] = data_df['acc_num'].astype(str)

    # required_cols = ["acc_num", "amt", "trans_num", "state", "zip", "city_pop", "unix_time", "ip_address"]

    # Feature engineering
    ip_trans_freq = data_df.groupby(["acc_num", "ip_address"])["trans_num"].count().reset_index()
    ip_trans_freq = ip_trans_freq.groupby("acc_num")["trans_num"].sum().reset_index()
    ip_trans_freq.columns = ["acc_num", "ip_trans_freq"]


    # Group by acc_num and aggregate features and use lambda to handle mode and max-min difference
    agg_features = data_df.groupby("acc_num").agg({
        "amt": ["mean", "std"],
        "trans_num": "count",
        "state": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown",
        "zip": "first",
        "city_pop": "mean",
        "unix_time": ["mean", lambda x: x.max() - x.min()],
        "ip_address": lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
    }).reset_index()

    # Flatten column names
    agg_features.columns = [
        "acc_num", "amt_mean", "amt_std", "trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_address"
    ]

    # Merge ip_trans_freq
    agg_features = agg_features.merge(ip_trans_freq, on="acc_num", how="left")
    agg_features["ip_trans_freq"] = agg_features["ip_trans_freq"].fillna(0)
    agg_features["amt_std"] = agg_features["amt_std"].fillna(0)

    # Encode categorical features
    agg_features["ip_address_encoded"] = agg_features["ip_address"].astype("category").cat.codes
    agg_features["state"] = agg_features["state"].astype("category").cat.codes
    agg_features["zip"] = agg_features["zip"].astype("category").cat.codes

    # Ensure trans_count is integer
    agg_features["trans_count"] = agg_features["trans_count"].astype(int)

    # Node feature matrix
    feature_cols = ["amt_mean", "amt_std", "trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_trans_freq", "ip_address_encoded"]
    
    x = torch.tensor(
        agg_features[feature_cols].values,
        dtype=torch.float
    )


    # Create a mapping from account numbers to indices
    acc_list = agg_features["acc_num"].tolist()
    acc_to_idx = {acc: idx for idx, acc in enumerate(acc_list)}
    # Create a reverse mapping from indices to account numbers
    idx_to_acc = {idx: acc for idx, acc in acc_to_idx.items()}

    # Build edges
    edges = set()

    # Group by zip and IP address
    zip_groups = data_df.groupby("zip")["acc_num"].apply(list)
    ip_groups = data_df.groupby("ip_address")["acc_num"].apply(list)


    # Add edges having the same IP address
    def add_ip_edges():
        for ip, nodes in ip_groups.items():
            if len(nodes) > 1:
                trans_counts = data_df[data_df["ip_address"] == ip].groupby("acc_num")["trans_num"].count()
                filtered_nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
                for src, dst in itertools.combinations(filtered_nodes, 2):
                    edges.add((acc_to_idx[src], acc_to_idx[dst]))
                    edges.add((acc_to_idx[dst], acc_to_idx[src]))


    # Add edges having the same zip code
    def add_zip_edges():
        for zip_code, nodes in zip_groups.items():
            if len(nodes) > 1:
                trans_counts = data_df[data_df["zip"] == zip_code].groupby("acc_num")["trans_num"].count()
                filtered_nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
                for src, dst in itertools.combinations(filtered_nodes, 2):
                    edges.add((acc_to_idx[src], acc_to_idx[dst]))
                    edges.add((acc_to_idx[dst], acc_to_idx[src]))


    # Add transaction edges based on time difference
    def add_transaction_edges():
        data_df_sorted = data_df.sort_values("unix_time")
        for ip, group in data_df_sorted.groupby("ip_address"):
            times = group["unix_time"].values
            nodes = group["acc_num"].values
            for i in range(len(times) - 1):
                if times[i + 1] - times[i] < 900:
                    src, dst = nodes[i], nodes[i + 1]
                    edges.add((acc_to_idx[src], acc_to_idx[dst]))
                    edges.add((acc_to_idx[dst], acc_to_idx[src]))

    add_ip_edges()
    add_zip_edges()
    add_transaction_edges()

    # Convert edges to tensor
    edge_index = torch.tensor(list(edges), dtype=torch.long).T

    # Create graph data object
    data = Data(x=x, edge_index=edge_index)

    # Load trained model
    model_path = "models/best_model.pt"
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' does not exist. Please run training script first.")
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
    
    try:
        out_channels = 216  # Verify with train_graphsage.py logs
        model = GraphSAGE(in_channels=x.size(1), hidden_channels=128, out_channels=out_channels)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # print(f"Loaded trained model from {model_path}.")
    except Exception as e:
        # print(f"Failed to load model: {e}")
        raise

    # Predict
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()

    # Extract fraud rings
    fraud_rings = defaultdict(list)
    for idx, pred_label in enumerate(pred):
        if pred_label != 0:
            fraud_rings[pred_label].append(idx_to_acc[idx])

    # Output format
    output_rings = [
        {
            "fraud_ring_id": int(pred_label),
            "accounts": sorted(fraud_rings[pred_label])
        }
        for pred_label in sorted(fraud_rings.keys())
    ]

    # Neo4j format
    neo4j_rings = [
        {
            "fraud_ring_id": int(pred_label),
            "accounts": [{"acc_num": str(acc)} for acc in sorted(fraud_rings[pred_label])]
        }
        for pred_label in sorted(fraud_rings.keys())
    ]

    # Update Neo4j
    # neo4j_conn = Neo4jConnection()

    # Save to JSON
    os.makedirs('output', exist_ok=True)
    output_path = "output/predicted_fraud_rings.json"

    with open(output_path, "w") as f:
        json.dump(output_rings, f, indent=2)
        print(f"Saved predicted fraud rings to '{output_path}'.")

    return output_rings

# Example usage
if __name__ == "__main__":
    transactions = [
        {'acc_num': 'acc123', 'amt': 100.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't1', 'unix_time': 1625097600, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc123', 'amt': 150.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't2', 'unix_time': 1625097601, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc456', 'amt': 200.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't3', 'unix_time': 1625097602, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc456', 'amt': 250.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't4', 'unix_time': 1625097603, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc789', 'amt': 300.0, 'state': 'CA', 'zip': '90001', 'city_pop': 500000, 'trans_num': 't5', 'unix_time': 1625097604, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc789', 'amt': 350.0, 'state': 'CA', 'zip': '90001', 'city_pop': 500000, 'trans_num': 't6', 'unix_time': 1625097605, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc123', 'amt': 1000.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't7', 'unix_time': 1625097606, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc123', 'amt': 2000.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't8', 'unix_time': 1625097607, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc456', 'amt': 1500.0, 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't9', 'unix_time': 1625097608, 'ip_address': '192.168.1.1'},
    ]

    predicted_rings = process_and_predict(transactions)

    # Print the predicted fraud rings
    print(json.dumps(predicted_rings, indent=2))