import os
from minio import Minio
from minio.error import S3Error
import tarfile
import logging
logging.basicConfig(level=logging.DEBUG)
import ssl


# Initialize MinIO client
client = Minio(
    "127.0.0.1:9001",
    access_key="ZcKMfjJZjF4bhyOuKBCM",
    secret_key="g02Ku5CSVk0YF0LcyQ52gVpAUr9M6EV1p3xmkYJG",
    secure=False  # Use True for HTTPS connection
)
####################################

# Create a custom SSL context
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE  # Be cautious with this in production

def extract_and_upload(bucket_name, object_name, target_bucket):
    try:
        # Download the tar.gz file
        response = client.get_object(bucket_name, object_name)
        with open('tempfile.tar.gz', 'wb') as file_data:
            for data in response.stream(32 * 1024):
                file_data.write(data)

        # Extract the tar.gz file
        with tarfile.open('tempfile.tar.gz', 'r:gz') as tar:
            tar.extractall(path='extracted_files')

        # Upload each file in the extracted directory
        for file_name in os.listdir('extracted_files'):
            file_path = os.path.join('extracted_files', file_name)
            client.fput_object(target_bucket, file_name, file_path)
            print(f"Uploaded {file_name} to {target_bucket}")

        # Cleanup local files
        os.remove('tempfile.tar.gz')
        for file_name in os.listdir('extracted_files'):
            os.remove(os.path.join('extracted_files', file_name))

    except S3Error as e:
        print("MinIO API error:", e)
    except Exception as e:
        print("General error:", e)


# Usage
extract_and_upload("targzfolder", "1438042981460.12.tar", "commondatacrawl")