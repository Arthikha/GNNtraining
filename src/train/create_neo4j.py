import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
# from collections import defaultdict
import itertools
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
import pickle

class Neo4jConnection:
    def __init__(self, uri=None, user=None, password=None, is_training=False):
        # if is_training:
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://neo4j-train:7687')
        self.user = user or os.getenv('NEO4J_USER', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'testpassword')
        
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.connected = True
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            self.connected = False

    def close(self):
        if self.driver:
            self.driver.close()

    def clear_database(self):
        if not self.connected:
            return
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_account_node(self, tx, acc_num, features):
        query = (
            '''MERGE (a:Account {acc_num: $acc_num})
               SET a.state = $state,
                   a.zip = $zip,
                   a.city_pop = $city_pop,
                   a.unix_time_mean = $unix_time_mean,
                   a.time_span = $time_span,
                   a.ip_trans_freq = $ip_trans_freq,
                   a.ip_address = $ip_address'''
        )
        features = dict(features)
        features["acc_num"] = acc_num
        features['ip_address'] = str(features['ip_address'])
        tx.run(query, **features)

    def create_edge(self, tx, src_acc, dst_acc, edge_type):
        query = (
            f"MATCH (a:Account {{acc_num: $src_acc}}), (b:Account {{acc_num: $dst_acc}}) "
            f"MERGE (a)-[:{edge_type}]->(b)"
        )
        tx.run(query, src_acc=src_acc, dst_acc=dst_acc)

    def create_fraud_ring(self, tx, ring_id, accounts):
        query = (
            '''MERGE (r:FraudRing {{ring_id: $ring_id}})
               WITH r
               UNWIND $accounts AS acc
               MATCH (a:Account {{acc_num: acc.acc_num}})
               MERGE (a)-[:IN_FRAUD_RING]->(r)'''
        )
        tx.run(query, ring_id=ring_id, accounts=accounts)

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

agg_features.columns = [
    "acc_num", "trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_address"
]

agg_features = agg_features.merge(ip_trans_freq, on="acc_num", how="left")
agg_features["ip_trans_freq"] = agg_features["ip_trans_freq"].fillna(0)

agg_features["ip_address_encoded"] = agg_features["ip_address"].astype("category").cat.codes
agg_features["state"] = agg_features["state"].astype("category").cat.codes

# Normalize numerical features
scaler = StandardScaler()
numerical_cols = ["city_pop", "unix_time_mean", "time_span", "ip_trans_freq"]
agg_features[numerical_cols] = scaler.fit_transform(agg_features[numerical_cols])

# Save scaler for prediction

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Node feature matrix
x = agg_features[["trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_trans_freq", "ip_address_encoded"]].values

# Build edges
edges = set()

def add_zip_ip_edges():
    combined_groups = data_df.groupby(["zip", "ip_address"])["node_index"].apply(list)
    for (zip_code, ip), nodes in combined_groups.items():
        if len(nodes) > 1:
            trans_counts = data_df[
                (data_df["zip"] == zip_code) & (data_df["ip_address"] == ip)
            ].groupby("node_index")["trans_num"].count()
            filtered_nodes = [n for n in nodes if trans_counts.get(n, 0) >= 1]
            for src, dst in itertools.combinations(filtered_nodes, 2):
                edges.add((src, dst))
                edges.add((dst, src))

def add_transaction_edges():
    data_df_sorted = data_df.sort_values("unix_time")
    for ip, group in data_df_sorted.groupby("ip_address"):
        times = group["unix_time"].values
        nodes = group["node_index"].values
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] < 900:
                src, dst = nodes[i], nodes[i + 1]
                edges.add((src, dst))
                edges.add((dst, src))

add_zip_ip_edges()
add_transaction_edges()

edges = list(edges)

# Encode labels
account_labels = data_df.groupby("acc_num")["fraud_ring_id"].first().reset_index()
account_labels["fraud_ring_id"] = account_labels["fraud_ring_id"].fillna(-1)  # Use -1 for non-fraud
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(account_labels["fraud_ring_id"])
# Map -1 (non-fraud) to label 0
y = np.where(y == label_encoder.transform([-1])[0], 0, y)

# Compute class weights
class_counts = np.bincount(y)
class_weights = 1.0 / class_counts
class_weights = np.clip(class_weights / class_weights.sum() * len(class_counts), 0.1, 5.0)

# Train-test split

train_indices, test_indices = train_test_split(range(len(unique_accounts)), test_size=0.2, random_state=42)
train_mask = np.zeros(len(unique_accounts), dtype=bool)
train_mask[train_indices] = True
test_mask = np.zeros(len(unique_accounts), dtype=bool)
test_mask[test_indices] = True

# Data object
data = type('Data', (), {
    'x': tf.convert_to_tensor(x, dtype=tf.float32),
    'y': tf.convert_to_tensor(y, dtype=tf.int32),
    'train_mask': train_mask,
    'test_mask': test_mask
})()

# Initialize Neo4j
neo4j_conn = Neo4jConnection(is_training=True)

def initialize_neo4j():
    if not neo4j_conn.connected:
        print("Neo4j connection not available")
        return
    
    neo4j_conn.clear_database()
    
    with neo4j_conn.driver.session() as session:
        for _, row in agg_features.iterrows():
            acc_num = row["acc_num"]
            features = {
                "state": float(row["state"]),
                "zip": float(row["zip"]),
                "city_pop": float(row["city_pop"]),
                "unix_time_mean": float(row["unix_time_mean"]),
                "time_span": float(row["time_span"]),
                "ip_trans_freq": float(row["ip_trans_freq"]),
                "ip_address": str(row["ip_address"])
            }
            session.write_transaction(neo4j_conn.create_account_node, acc_num, features)
        for src, dst in edges:
            src_acc = idx_to_acc[src]
            dst_acc = idx_to_acc[dst]
            session.execute_write(lambda tx: neo4j_conn.create_edge(tx, src_acc, dst_acc, "RELATED"))
    
    print("Neo4j training database initialized.")

if __name__ == '__main__':
    initialize_neo4j()
    neo4j_conn.close()