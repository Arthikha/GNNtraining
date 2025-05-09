# import json
# from kafka import KafkaConsumer
# from neo4j import GraphDatabase
# import torch
# import dgl
# import networkx as nx

# # === Neo4j Setup ===
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PASS = "your_password"

# # === Kafka Setup ===
# KAFKA_TOPIC = "transactions"
# KAFKA_BOOTSTRAP_SERVERS = ["localhost:9092"]

# # === GNN Model (assume loaded GraphSAGE) ===
# from mymodel import TrainedGraphSAGE, build_dgl_graph_from_neo4j  # You must implement this

# model = TrainedGraphSAGE()
# model.load_state_dict(torch.load("model.pth"))
# model.eval()

# # === Neo4j Driver ===
# driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# def add_transaction_to_neo4j(tx_data):
#     with driver.session() as session:
#         session.write_transaction(_add_txn, tx_data)

# def _add_txn(tx, tx_data):
#     # Example: { "from": "acc123", "to": "acc456", "amount": 300 }
#     query = """
#     MERGE (a:Account {acc_num: $from})
#     MERGE (b:Account {acc_num: $to})
#     MERGE (a)-[:TRANSFERRED_TO {amount: $amount}]->(b)
#     """
#     tx.run(query, from=tx_data["from"], to=tx_data["to"], amount=tx_data["amount"])

# # === Inference Pipeline ===
# def run_inference_and_group_rings():
#     g, node_ids = build_dgl_graph_from_neo4j(driver)
#     with torch.no_grad():
#         logits = model(g)
#         fraud_rings = torch.argmax(logits, dim=1).tolist()

#     acc_fraud_map = {}
#     for acc_id, ring_id in zip(node_ids, fraud_rings):
#         acc_fraud_map.setdefault(f"fraud_ring{ring_id}", []).append(acc_id)

#     return acc_fraud_map

# # === Kafka Consumer Loop ===
# consumer = KafkaConsumer(
#     KAFKA_TOPIC,
#     bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
#     auto_offset_reset="latest",
#     value_deserializer=lambda x: json.loads(x.decode("utf-8")),
# )

# print("Listening to Kafka topic...")

# for msg in consumer:
#     tx_data = msg.value
#     print(f"Received: {tx_data}")
    
#     # Step 1: Add to Neo4j
#     add_transaction_to_neo4j(tx_data)

#     # Step 2: Run inference
#     fraud_rings = run_inference_and_group_rings()

#     # Step 3: Output
#     for ring, accounts in fraud_rings.items():
#         print(f"{ring}: {accounts}")


