import pandas as pd
import random

# Load your dataset
df = pd.read_csv("fraud_with_rings.csv")

# Function to generate a fake but consistent IP address from zip code
def generate_ip(zipcode):
    random.seed(int(zipcode))  # ensures same zip gets same IP
    return f"192.168.{int(zipcode) % 256}.{random.randint(1, 254)}"

# Add the IP address column
df['ip_address'] = df['zip'].apply(generate_ip)

# Save back to a new CSV
df.to_csv("fraud_with_rings_2.csv", index=False)

print("✅ IP addresses added and saved to fraud_with_ips.csv")
