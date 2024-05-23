'''import os
import gzip
import pandas as pd
from minio import Minio
from minio.error import S3Error
import ssl
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Minio Settings
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
'''
import time

###################################################################################################################################

#'''
import os
import json
import random
import re

import pandas as pd
from minio import Minio
from minio.error import S3Error
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Minio Settings
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False  # Set to True for HTTPS
)


def read_json_lines(filepath):
    encodings = ['utf-8', 'latin1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error("JSON decode error in line with %s encoding: %s", encoding, e)
                        continue
        except UnicodeDecodeError:
            logging.warning("Failed decoding with %s, trying next...", encoding)
            continue
    raise Exception("All encoding attempts failed.")


def transpose_relation_data(relation_data):
    # Transpose the columns to rows
    transposed_data = list(zip(*relation_data))
    return transposed_data


def process_json_to_csv(bucket_name, target_bucket):
    logging.info("Listing objects in bucket: %s", bucket_name)
    objects = client.list_objects(bucket_name, recursive=True)
    i = 0
    j = 105
    k = 0
    for obj in objects:
        if obj.object_name.endswith('.json'):
            local_json_path = os.path.join('/tmp', obj.object_name)
            try:
                logging.info("Processing JSON file: %s", obj.object_name)
                # Download the JSON file
                response = client.get_object(bucket_name, obj.object_name)
                with open(local_json_path, 'wb') as file_data:
                    for d in response.stream(32 * 1024):
                        file_data.write(d)

                # Read JSON lines from the file
                for data in read_json_lines(local_json_path):
                    if data is None:
                        continue

                    # Debugging: Log the first part of the data to ensure it's loaded correctly
                    logging.debug(f"Loaded JSON data: {str(data)[:500]}...")

                    # Extract pageTitle for naming the CSV files
                    page_title = data.get('pageTitle', 'unknown_page').replace(' ', '_')

                    # Process the relation data
                    relation_data = data.get('relation', [])
                    meta_data = data.get('url')
                    if relation_data:
                        # Transpose the relation data to convert columns to rows
                        transposed_data = transpose_relation_data(relation_data)

                        # Convert the transposed data to a DataFrame
                        try:
                            json_df = pd.DataFrame(transposed_data[1:], columns=transposed_data[0])
                        except Exception as e:
                            logging.error(f"Failed to convert relation data to DataFrame: {e}")
                            continue

                        # Define the CSV file path
                        if not page_title:
                            page_title = 'randomName' + random.Random()
                        filtered_path = page_title.replace("/", "")
                        filtered_path = re.sub(r'[^a-zA-Z0-9/-]', '', filtered_path)
                        if not filtered_path:
                            filtered_path = 'randomName' + random.Random()
                        # Limit the length to 25 characters
                        filtered_path = filtered_path[:60]
                        csv_file_name = f"{filtered_path}.csv"
                        csv_file_path = local_json_path + '.csv'

                        # Save DataFrame to CSV
                        try:
                            json_df.to_csv(csv_file_path, index=False)
                            logging.info("Converted JSON table to CSV and saved locally: %s", csv_file_path)

                            # Get the current prefix based on the number of files
                            #current_prefix = get_current_prefix(target_bucket, max_files_per_prefix)

                            '''
                            # Full object name with prefix
                            i = i + 1
                            if i >= 198:
                                j = j + 1
                                i = 0
                                time.sleep(4)
                                if j >= 198:
                                    k = k + 1
                                    j = 0
                                    i = 0
                                    time.sleep(4)
                            object_name = f"part{k}/part{j}/{csv_file_name}"
                            '''

                            # Upload CSV file to MinIO with the calculated prefix
                            client.fput_object(target_bucket, csv_file_name, csv_file_path)
                            logging.info("Uploaded CSV to MinIO bucket: %s with prefix %s", target_bucket)
                        except Exception as e:
                            logging.error(f"Failed to save or upload CSV file {csv_file_name}: {e}")
                            continue
                    else:
                        logging.warning("No relation data found in JSON file: %s", obj.object_name)

            except S3Error as e:
                logging.error("MinIO API error while processing %s: %s", obj.object_name, e)
            except FileNotFoundError as e:
                logging.error("File not found error while processing %s: %s", obj.object_name, e)
            except Exception as e:
                logging.error("General error while processing %s: %s", obj.object_name, e)
            finally:
                # Cleanup local files
                if os.path.exists(local_json_path):
                    os.remove(local_json_path)
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)


# Usage

if __name__ == "__main__":
    process_json_to_csv("1438042981856.5", "commonwebtables")

print("Ende")

#'''


##################################################################################################################################
'''
import os
import json
import random
import time
import pandas as pd
from minio import Minio
from minio.error import S3Error
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Minio Settings
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False  # Set to True for HTTPS
)


def get_initial_counts(bucket_name, base_prefix):
    objects = client.list_objects(bucket_name, prefix=base_prefix, recursive=True)
    file_counts = {}
    for obj in objects:
        prefix = obj.object_name.split('/')[0]
        if prefix not in file_counts:
            file_counts[prefix] = 0
        file_counts[prefix] += 1
    return file_counts


def get_next_prefix(file_counts, base_prefix, max_files_per_prefix):
    for i in range(len(file_counts)):
        if file_counts.get(f'{base_prefix}part{i}', 0) < max_files_per_prefix:
            return f'{base_prefix}part{i}/', file_counts.get(f'{base_prefix}part{i}', 0)
    new_prefix = f'{base_prefix}part{len(file_counts)}/'
    file_counts[new_prefix] = 0
    return new_prefix, 0


def read_json_lines(filepath):
    encodings = ['utf-8', 'latin1', 'windows-1252']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as e:
                        logging.error("JSON decode error in line with %s encoding: %s", encoding, e)
                        continue
        except UnicodeDecodeError:
            logging.warning("Failed decoding with %s, trying next...", encoding)
            continue
    raise Exception("All encoding attempts failed.")


def transpose_relation_data(relation_data):
    transposed_data = list(zip(*relation_data))
    return transposed_data


def process_json_to_csv(bucket_name, target_bucket, base_prefix, max_files_per_prefix):
    logging.info("Listing objects in bucket: %s", bucket_name)
    objects = client.list_objects(bucket_name, recursive=True)

    file_counts = get_initial_counts(target_bucket, base_prefix)
    current_prefix, current_count = get_next_prefix(file_counts, base_prefix, max_files_per_prefix)

    for obj in objects:
        if obj.object_name.endswith('.json'):
            local_json_path = os.path.join('/tmp', obj.object_name)
            try:
                logging.info("Processing JSON file: %s", obj.object_name)
                response = client.get_object(bucket_name, obj.object_name)
                with open(local_json_path, 'wb') as file_data:
                    for d in response.stream(32 * 1024):
                        file_data.write(d)

                for data in read_json_lines(local_json_path):
                    if data is None:
                        continue

                    logging.debug(f"Loaded JSON data: {str(data)[:500]}...")

                    page_title = data.get('pageTitle', 'unknown_page').replace(' ', '_')
                    relation_data = data.get('relation', [])
                    meta_data = data.get('url')
                    if relation_data:
                        transposed_data = transpose_relation_data(relation_data)
                        try:
                            json_df = pd.DataFrame(transposed_data[1:], columns=transposed_data[0])
                        except Exception as e:
                            logging.error(f"Failed to convert relation data to DataFrame: {e}")
                            continue

                        if not page_title:
                            page_title = 'randomName' + random.Random()
                        filtered_path = page_title.replace("/", "")
                        csv_file_name = f"{filtered_path}.csv"
                        csv_file_path = local_json_path + '.csv'

                        try:
                            json_df.to_csv(csv_file_path, index=False)
                            logging.info("Converted JSON table to CSV and saved locally: %s", csv_file_path)

                            client.fput_object(target_bucket, current_prefix + csv_file_name, csv_file_path)
                            logging.info("Uploaded CSV to MinIO bucket: %s with prefix %s", target_bucket,
                                         current_prefix)

                            current_count += 1
                            file_counts[current_prefix] = current_count
                            if current_count >= max_files_per_prefix:
                                time.sleep(2)
                                current_prefix, current_count = get_next_prefix(file_counts, base_prefix,
                                                                                max_files_per_prefix)
                        except Exception as e:
                            logging.error(f"Failed to save or upload CSV file {csv_file_name}: {e}")
                            continue
                    else:
                        logging.warning("No relation data found in JSON file: %s", obj.object_name)

            except S3Error as e:
                logging.error("MinIO API error while processing %s: %s", obj.object_name, e)
            except FileNotFoundError as e:
                logging.error("File not found error while processing %s: %s", obj.object_name, e)
            except Exception as e:
                logging.error("General error while processing %s: %s", obj.object_name, e)
            finally:
                if os.path.exists(local_json_path):
                    os.remove(local_json_path)
                if os.path.exists(csv_file_path):
                    os.remove(csv_file_path)


if __name__ == "__main__":
    process_json_to_csv("testrun2", "commondatacrawl2", "part", 200)

print("Ende")

'''