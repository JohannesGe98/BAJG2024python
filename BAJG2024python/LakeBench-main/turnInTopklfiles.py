import pandas as pd
import os
import pickle

# Directory containing the CSV files
csv_directory = '/Users/johannesgesk/Documents_MacIntouch/Repositories/BAJG2024python/commonwebcrawlerhuge'

# Dictionary to hold DataFrames
dataframes ={}

# Loop through the CSV files in the directory
for filename in os.listdir(csv_directory):
    if filename.endswith('.csv'):
        # Read the CSV file into a DataFrame
        file_path = os.path.join(csv_directory, filename)
        df = pd.read_csv(file_path)

        # Use the filename (without extension) as the key
        key = os.path.splitext(filename)[0]
        dataframes[key] = df

# Serialize the dictionary into a PKL file
with open('commonwebcrawlerhuge.pkl', 'wb') as pkl_file:
    pickle.dump(dataframes, pkl_file)

print("All CSV files have been stored in dataframes.pkl")