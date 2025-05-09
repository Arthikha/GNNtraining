import os
import pandas as pd
import numpy as np
import tensorflow as tf
import dgl
from dgl.nn.tensorflow import SAGEConv
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import json
from .create_neo4j import data, y, class_weights, agg_features, edges, idx_to_acc, label_encoder

# Set random seed
tf.random.set_seed(42)
np.random.seed(42)

# Convert data to DGL graph
def create_dgl_graph():
    src, dst = zip(*edges)
    g = dgl.graph((src, dst))
    g.ndata['feat'] = tf.convert_to_tensor(agg_features[["trans_count", "state", "zip", "city_pop", "unix_time_mean", "time_span", "ip_trans_freq", "ip_address_encoded"]].values, dtype=tf.float32)
    g.ndata['label'] = tf.convert_to_tensor(y, dtype=tf.int32)
    g.ndata['train_mask'] = tf.convert_to_tensor(data.train_mask, dtype=tf.bool)
    return g

graph = create_dgl_graph()

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

# Initialize model
in_feats = graph.ndata['feat'].shape[1]
out_feats = int(tf.reduce_max(y).numpy() + 1)
model = GraphSAGE(in_feats, hidden_feats=128, out_feats=out_feats)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, weight_decay=1e-4)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, weight=class_weights)

# Compile model
model.compile(optimizer=optimizer, loss=loss_fn)

# Training
@tf.function
def train_step(g, features, labels, mask):
    with tf.GradientTape() as tape:
        logits = model(g, features)
        loss = loss_fn(labels[mask], logits[mask])
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# Training loop with early stopping
best_loss = float('inf')
patience = 30
epochs_without_improvement = 0
model_path = "data/models"

for epoch in range(1, 201):
    loss = train_step(graph, graph.ndata['feat'], graph.ndata['label'], graph.ndata['train_mask'])
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

# Predict fraud rings for training data
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

print("Training completed. Model saved to data/models/. Fraud rings saved to data/train_fraud_rings.json.")

print("\nRun this Cypher query in Neo4j to visualize fraud rings:")
print("MATCH (a:Account)-[r:IN_FRAUD_RING]->(f:FraudRing) RETURN a, r, f LIMIT 50")
print("Neo4j credentials: user=neo4j, password=testpassword, URI=bolt://neo4j:7687")

# Close Neo4j connection
# neo4j_conn.close()