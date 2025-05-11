import os
import pandas as pd
import numpy as np
import tensorflow as tf
import dgl
from dgl.nn.tensorflow import SAGEConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
import itertools
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
import json

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Neo4jConnection class for database operations
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://neo4j-train:7687')
        self.user = os.getenv('NEO4J_USER', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'testpassword')
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

# Load and preprocess dataset
def load_and_preprocess_data():
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

    # Encode categorical features
    agg_features["ip_address_encoded"] = agg_features["ip_address"].astype("category").cat.codes
    agg_features["state"] = agg_features["state"].astype("category").cat.codes

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ["city_pop", "unix_time_mean", "time_span", "ip_trans_freq"]
    agg_features[numerical_cols] = scaler.fit_transform(agg_features[numerical_cols])

    # Save scaler for prediction
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    return data_df, agg_features, acc_to_idx, idx_to_acc

# Build edges for the graph
def build_edges(data_df, acc_to_idx):
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
    return list(edges)

# Encode labels and compute class weights
def encode_labels_and_compute_weights(data_df, unique_accounts):
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

    return y, class_weights

# Create train-test split
def create_train_test_split(unique_accounts):
    train_indices, test_indices = train_test_split(range(len(unique_accounts)), test_size=0.2, random_state=42)
    train_mask = np.zeros(len(unique_accounts), dtype=bool)
    train_mask[train_indices] = True
    test_mask = np.zeros(len(unique_accounts), dtype=bool)
    test_mask[test_indices] = True
    return train_mask, test_mask

# Create DGL graph
def create_dgl_graph(edges, agg_features, y, train_mask):
    src, dst = zip(*edges)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = tf.convert_to_tensor(
        agg_features[["trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_trans_freq", "ip_address_encoded"]].values,
        dtype=tf.float32
    )
    g.ndata['label'] = tf.convert_to_tensor(y, dtype=tf.int32)
    g.ndata['train_mask'] = tf.convert_to_tensor(train_mask, dtype=tf.bool)
    return g

# Initialize Neo4j database
def initialize_neo4j(neo4j_conn, agg_features, edges, idx_to_acc):
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

# Define GraphSAGE model
class GraphSAGE(tf.keras.Model):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout_rate=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_feats, hidden_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, g, features):
        h = self.conv1(g, features)
        h = tf.nn.relu(h)
        h = self.dropout(h)
        h = self.conv2(g, h)
        return h

# Training loop with early stopping
def train_model(model, graph, loss_fn, optimizer, epochs=200, patience=30):
    best_loss = float('inf')
    epochs_without_improvement = 0
    model_path = "models"

    for epoch in range(1, epochs + 1):
        loss = train_step(graph, graph.ndata['feat'], graph.ndata['label'], graph.ndata['train_mask'], model, loss_fn, optimizer)
        loss_value = loss.numpy()
        if loss_value < best_loss:
            best_loss = loss_value
            epochs_without_improvement = 0
            # Save the entire model
            tf.keras.models.save_model(model, model_path, save_format='tf')
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    return model

@tf.function
def train_step(g, features, labels, mask, model, loss_fn, optimizer):
    with tf.GradientTape() as tape:
        logits = model(g, features)
        loss = loss_fn(labels[mask], logits[mask])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Predict fraud rings and save results
def predict_and_save_fraud_rings(model, graph, agg_features):
    logits = model(graph, graph.ndata['feat'])
    pred = tf.argmax(logits, axis=1).numpy()

    fraud_rings = defaultdict(list)
    acc_nums = agg_features['acc_num'].values
    for idx, pred_label in enumerate(pred):
        fraud_rings[pred_label].append(acc_nums[idx])

    train_rings = []
    for pred_label, acc_list in sorted(fraud_rings.items(), key=lambda x: x[0]):
        if pred_label != 0:  # Exclude non-fraud (label 0)
            train_rings.append({
                'fraud_ring_id': pred_label,
                'accounts': sorted(acc_list)
            })
        else:
            # Non-fraud accounts
            for acc in acc_list:
                train_rings.append({
                    'fraud_ring_id': 0,
                    'accounts': [acc]
                })

    # Save fraud ring assignments
    with open('data/train_fraud_rings.json', 'w') as f:
        json.dump(train_rings, f, indent=2)

# Main execution
if __name__ == '__main__':
    # Initialize Neo4j connection
    neo4j_conn = Neo4jConnection()

    # Load and preprocess data
    data_df, agg_features, acc_to_idx, idx_to_acc = load_and_preprocess_data()

    # Build edges
    edges = build_edges(data_df, acc_to_idx)

    # Encode labels and compute class weights
    y, class_weights = encode_labels_and_compute_weights(data_df, agg_features['acc_num'].unique())

    # Create train-test split
    train_mask, test_mask = create_train_test_split(agg_features['acc_num'].unique())

    # Create DGL graph
    graph = create_dgl_graph(edges, agg_features, y, train_mask)

    # Initialize Neo4j database
    initialize_neo4j(neo4j_conn, agg_features, edges, idx_to_acc)

    # Initialize and compile GraphSAGE model
    in_feats = graph.ndata['feat'].shape[1]
    out_feats = int(tf.reduce_max(y).numpy() + 1)
    model = GraphSAGE(in_feats, hidden_feats=128, out_feats=out_feats)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, weight_decay=1e-4)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, weight=class_weights)
    model.compile(optimizer=optimizer, loss=loss_fn)

    # Train the model
    model = train_model(model, graph, loss_fn, optimizer)

    # Predict fraud rings and save results
    predict_and_save_fraud_rings(model, graph, agg_features)

    # Print completion message
    print("Training completed. Model saved to models/. Fraud rings saved to data/train_fraud_rings.json.")
    print("\nRun this Cypher query in Neo4j to visualize fraud rings:")
    print("MATCH (a:Account)-[r:IN_FRAUD_RING]->(f:FraudRing) RETURN a, r, f LIMIT 50")
    print(f"Neo4j credentials: user={neo4j_conn.user}, password={neo4j_conn.password}, URI={neo4j_conn.uri}")

    # Close Neo4j connection
    neo4j_conn.close()
