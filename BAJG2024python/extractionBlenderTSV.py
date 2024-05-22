import os
import pandas as pd
from minio import Minio
from minio.error import S3Error
import logging
import csv  # Ensure the csv module is imported
import time  # For timing purposes

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Minio Settings
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False  # Set to True for HTTPS
)


def process_large_tsv_to_csv(bucket_name, target_bucket, max_csv_files=23000):
    logging.info("Listing objects in bucket: %s", bucket_name)
    objects = client.list_objects(bucket_name, recursive=True)
    csv_file_count = 0

    for obj in objects:
        if obj.object_name.endswith('.tsv'):
            if csv_file_count >= max_csv_files:
                logging.info("Reached the maximum limit of %d CSV files. Stopping further processing.", max_csv_files)
                break

            local_tsv_path = os.path.join('/tmp', obj.object_name)
            try:
                logging.info("Processing TSV file: %s", obj.object_name)
                start_time = time.time()
                # Download the TSV file
                response = client.get_object(bucket_name, obj.object_name)
                with open(local_tsv_path, 'wb') as file_data:
                    for d in response.stream(32 * 1024):
                        file_data.write(d)
                logging.info("Downloaded TSV file: %s in %.2f seconds", obj.object_name, time.time() - start_time)

                # Read TSV file
                df = pd.read_csv(local_tsv_path, sep='\t', quoting=csv.QUOTE_NONE, encoding='utf-8')

                # Process each row separately
                for index, row in df.iterrows():
                    if csv_file_count >= max_csv_files:
                        break

                    # Use the value from the 3rd column as the filename
                    csv_file_name = f"{row[2]}.csv"
                    csv_file_path = os.path.join('/tmp', csv_file_name)

                    try:
                        # Save the single row to a CSV file
                        row.to_frame().T.to_csv(csv_file_path, index=False)
                        logging.info("Converted TSV row to CSV and saved locally: %s", csv_file_path)

                        # Upload CSV file to MinIO
                        start_time = time.time()
                        client.fput_object(target_bucket, csv_file_name, csv_file_path)
                        logging.info("Uploaded CSV to MinIO bucket: %s in %.2f seconds", target_bucket,
                                     time.time() - start_time)
                        csv_file_count += 1

                    except Exception as e:
                        logging.error(f"Failed to save or upload CSV file {csv_file_path}: {e}")
                        continue
                    finally:
                        # Cleanup local CSV file
                        if os.path.exists(csv_file_path):
                            os.remove(csv_file_path)

            except S3Error as e:
                logging.error("MinIO API error while processing %s: %s", obj.object_name, e)
            except FileNotFoundError as e:
                logging.error("File not found error while processing %s: %s", obj.object_name, e)
            except Exception as e:
                logging.error("General error while processing %s: %s", obj.object_name, e)
            finally:
                # Cleanup local TSV file
                if os.path.exists(local_tsv_path):
                    os.remove(local_tsv_path)


# Usage
if __name__ == "__main__":
    process_large_tsv_to_csv("trec20000", "trectables")

print("Ende")