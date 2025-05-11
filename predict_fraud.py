import pandas as pd
import numpy as np
import tensorflow as tf
import dgl
from dgl.nn.tensorflow import SAGEConv
from collections import defaultdict
import json
import os
from neo4j import GraphDatabase

# Neo4jConnection class
class Neo4jConnection:
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://neo4j-train:7687')
        self.user =os.getenv('NEO4J_USER', 'neo4j')
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

    def create_account_node(self, tx, acc_num, features):
        query = """
        MERGE (a:Account {acc_num: $acc_num})
        SET a += $features
        """
        tx.run(query, acc_num=acc_num, features=features)

    def create_edge(self, tx, src, dst, rel_type):
        query = """
        MATCH (a1:Account {acc_num: $src}), (a2:Account {acc_num: $dst})
        MERGE (a1)-[r:RELATED]->(a2)
        """
        tx.run(query, src=src, dst=dst)

    def create_fraud_ring(self, tx, ring_id, accounts):
        query = """
        MERGE (f:FraudRing {ring_id: $ring_id})
        WITH f
        UNWIND $accounts AS acc
        MATCH (a:Account {acc_num: acc.acc_num})
        MERGE (a)-[:IN_FRAUD_RING]->(f)
        """
        tx.run(query, ring_id=ring_id, accounts=accounts)

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# Neo4j connection
neo4j_conn = Neo4jConnection()

# Preprocess transactions into account-level features
def preprocess_transactions(transactions):
    if not isinstance(transactions, pd.DataFrame):
        transactions = pd.DataFrame(
            transactions,
            columns=['acc_num', 'state', 'zip', 'city_pop', 'trans_num', 'unix_time', 'ip_address']
        )

    # Aggregate features per account
    agg_features = transactions.groupby('acc_num').agg({
        'state': lambda x: x.mode()[0] if not x.empty else 'unknown',
        'zip': lambda x: x.mode()[0] if not x.empty else 0,
        'city_pop': 'mean',
        'trans_num': 'count',
        'unix_time': ['mean', lambda x: x.max() - x.min()],  # unix_time_mean, time_span
        'ip_address': ['count', lambda x: x.mode()[0]]  # ip_trans_freq, ip_address
    }).reset_index()

    # Flatten multi-level columns
    agg_features.columns = [
        'acc_num', 'state', 'zip', 'city_pop', 'trans_count',
        'unix_time_mean', 'time_span', 'ip_trans_freq', 'ip_address'
    ]

    # Convert categorical features to numeric
    agg_features['state'] = pd.Categorical(agg_features['state']).codes
    agg_features['ip_address_encoded'] = pd.Categorical(agg_features['ip_address']).codes
    agg_features = agg_features.drop('ip_address', axis=1)

    # Ensure feature columns match training
    expected_columns = [
        'acc_num', 'trans_count', 'state', 'zip', 'city_pop',
        'unix_time_mean', 'time_span', 'ip_trans_freq', 'ip_address_encoded'
    ]
    for col in expected_columns:
        if col not in agg_features.columns:
            agg_features[col] = 0
    agg_features = agg_features[expected_columns]

    return agg_features

# Infer edges between accounts
def infer_edges(transactions, agg_features):
    edges = []
    acc_to_idx = {acc: idx for idx, acc in enumerate(agg_features['acc_num'])}
    
    for _, group in transactions.groupby('ip_address'):
        acc_nums = group['acc_num'].unique()
        for i in range(len(acc_nums)):
            for j in range(i + 1, len(acc_nums)):
                src = acc_to_idx.get(acc_nums[i])
                dst = acc_to_idx.get(acc_nums[j])
                if src is not None and dst is not None:
                    edges.append((src, dst))
                    edges.append((dst, src))

    for _, group in transactions.groupby('zip'):
        acc_nums = group['acc_num'].unique()
        for i in range(len(acc_nums)):
            for j in range(i + 1, len(acc_nums)):
                src = acc_to_idx.get(acc_nums[i])
                dst = acc_to_idx.get(acc_nums[j])
                if src is not None and dst is not None:
                    edges.append((src, dst))
                    edges.append((dst, src))

    return edges

# Create DGL graph
def create_dgl_graph(agg_features, edges):
    src, dst = zip(*edges) if edges else ([], [])
    g = dgl.graph((src, dst))
    g.ndata['feat'] = tf.convert_to_tensor(
        agg_features.drop('acc_num', axis=1).values, dtype=tf.float32
    )
    return g

# Predict fraud rings and assign new fraud_ring_id
def get_fraud_rings(model, graph, agg_features):
    logits = model(graph, graph.ndata['feat'])
    pred = tf.argmax(logits, axis=1).numpy()

    fraud_rings = defaultdict(list)
    acc_nums = agg_features['acc_num'].values
    for idx, pred_label in enumerate(pred):
        if pred_label != 0:  # Exclude non-fraud
            fraud_rings[pred_label].append(acc_nums[idx])

    # Assign new fraud_ring_id starting from 101
    output_rings = []
    neo4j_rings = []
    for ring_id, (pred_label, acc_list) in enumerate(
        sorted(fraud_rings.items(), key=lambda x: x[0]), start=101
    ):
        output_rings.append({
            'fraud_ring_id': ring_id,
            'accounts': sorted(acc_list)
        })
        neo4j_rings.append({
            'fraud_ring_id': ring_id,
            'accounts': [{'acc_num': acc} for acc in sorted(acc_list)]
        })

    return output_rings, neo4j_rings

# Update Neo4j incrementally
def update_neo4j(agg_features, edges, neo4j_rings):
    if not neo4j_conn.connected:
        print("Neo4j connection not available")
        return

    with neo4j_conn.driver.session() as session:
        # Add/update accounts
        for _, row in agg_features.iterrows():
            acc_num = row['acc_num']
            features = row.drop('acc_num').to_dict()
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (a:Account {acc_num: $acc_num})
                    SET a += $features
                    """,
                    acc_num=acc_num, features=features
                )
            )

        # Add edges
        for src, dst in edges:
            src_acc = agg_features['acc_num'].iloc[src]
            dst_acc = agg_features['acc_num'].iloc[dst]
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MATCH (a1:Account {acc_num: $src_acc}), (a2:Account {acc_num: $dst_acc})
                    MERGE (a1)-[:RELATED]->(a2)
                    """,
                    src_acc=src_acc, dst_acc=dst_acc
                )
            )

        # Add fraud rings
        for ring in neo4j_rings:
            ring_id = ring['fraud_ring_id']
            accounts = ring['accounts']
            session.execute_write(
                lambda tx: tx.run(
                    """
                    MERGE (f:FraudRing {ring_id: $ring_id})
                    WITH f
                    UNWIND $accounts AS acc
                    MATCH (a:Account {acc_num: acc.acc_num})
                    MERGE (a)-[:IN_FRAUD_RING]->(f)
                    """,
                    ring_id=ring_id, accounts=accounts
                )
            )

# Main prediction function
def predict_fraud(transactions):
    # Preprocess transactions
    agg_features = preprocess_transactions(transactions)
    
    # Infer edges
    edges = infer_edges(transactions, agg_features)
    
    # Create DGL graph
    graph = create_dgl_graph(agg_features, edges)
    
    # Load trained model
    try:
        model = tf.keras.models.load_model(
            'models', custom_objects={'SAGEConv': SAGEConv}
        )
    except Exception as e:
        print(f"Error: Failed to load model from 'models': {e}")
        return []
    
    # Predict fraud rings
    output_rings, neo4j_rings = get_fraud_rings(model, graph, agg_features)
    
    # Update Neo4j
    update_neo4j(agg_features, edges, neo4j_rings)
    
    # Save to JSON
    with open('predicted_fraud_rings.json', 'w') as f:
        json.dump(output_rings, f, indent=2)
    
    return output_rings

# Example usage
if __name__ == '__main__':
    transactions = [
        {'acc_num': 'acc123', 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't1', 'unix_time': 1625097600, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc456', 'state': 'NY', 'zip': '10001', 'city_pop': 1000000, 'trans_num': 't2', 'unix_time': 1625097601, 'ip_address': '192.168.1.1'},
        {'acc_num': 'acc789', 'state': 'CA', 'zip': '90001', 'city_pop': 500000, 'trans_num': 't3', 'unix_time': 1625097602, 'ip_address': '192.168.1.2'}
    ]
    
    fraud_rings = predict_fraud(transactions)
    print(json.dumps(fraud_rings, indent=2))
    
    # Close Neo4j connection
    neo4j_conn.close()
