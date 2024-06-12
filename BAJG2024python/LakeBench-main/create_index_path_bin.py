import pickle
import hnswlib
import numpy as np

# Load the tables and queries
table_path = "/LakeBench-main/commonwebcrawlerhuge.pkl"
query_path = "/dataframes_query.pkl"

with open(table_path, 'rb') as f:
    tables = pickle.load(f)

with open(query_path, 'rb') as f:
    queries = pickle.load(f)

# Function to encode tables into vectors
def encode_table(table):
    # Implement your table encoding logic here
    # For simplicity, we return a random vector (replace this with actual encoding)
    return np.random.rand(128)

# Encode all tables into vectors
table_vectors = [encode_table(table) for table in tables]

# Initialize the HNSW index
dim = 128  # Dimensionality of the vectors
num_elements = len(table_vectors)
hnsw_index = hnswlib.Index(space='cosine', dim=dim)
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16)

# Insert the encoded vectors into the HNSW index
for idx, vector in enumerate(table_vectors):
    hnsw_index.add_items(vector, idx)

# Save the index to a binary file
index_path = ".hnsw_index.bin"
hnsw_index.save_index(index_path)
