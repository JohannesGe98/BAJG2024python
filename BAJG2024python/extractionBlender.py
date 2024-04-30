import os
import gzip
import pandas as pd
from minio import Minio
from minio.error import S3Error
import ssl
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize MinIO client
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False  # Set to True for HTTPS
)


def process_json_gz_to_csv(bucket_name, target_bucket):
    logging.info("Listing objects in bucket: %s", bucket_name)
    objects = client.list_objects(bucket_name, recursive=True)

    for obj in objects:
        if obj.object_name.endswith('.json.gz'):
            local_gz_path = os.path.join('/tmp', obj.object_name)
            try:
                logging.info("Processing JSON.gz file: %s", obj.object_name)
                # Download the JSON.gz file
                response = client.get_object(bucket_name, obj.object_name)
                with open(local_gz_path, 'wb') as file_data:
                    for d in response.stream(32 * 1024):
                        file_data.write(d)

                # Attempt to read the file with different encodings
                encodings = ['utf-8', 'latin1', 'windows-1252']
                for encoding in encodings:
                    try:
                        with gzip.open(local_gz_path, 'rt', encoding=encoding) as f:
                            json_df = pd.read_json(f, lines=True)
                        break  # If reading succeeds, break out of the encoding loop
                    except UnicodeDecodeError:
                        logging.warning("Failed decoding with %s, trying next...", encoding)
                        continue
                else:
                    logging.error("All encoding attempts failed.")
                    continue

                # Convert DataFrame to CSV
                csv_file_path = local_gz_path[:-3] + '.csv'
                json_df.to_csv(csv_file_path, index=False)
                logging.info("Converted JSON to CSV and saved locally: %s", csv_file_path)

                # Upload CSV file to MinIO
                client.fput_object(target_bucket, os.path.basename(csv_file_path), csv_file_path)
                logging.info("Uploaded CSV to MinIO bucket: %s", target_bucket)

            except S3Error as e:
                logging.error("MinIO API error while processing %s: %s", obj.object_name, e)
            except Exception as e:
                logging.error("General error while processing %s: %s", obj.object_name, e)
            finally:
                # Cleanup local files
                if os.path.exists(local_gz_path):
                    os.remove(local_gz_path)
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)


# Usage
if __name__ == "__main__":
    process_json_gz_to_csv("buckettargzsource", "commondatacrawl")
