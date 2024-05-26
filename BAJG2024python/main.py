import json
import logging
import os
import re
import ssl
import string
import time

import aiohttp
import nltk as nltk
import numpy as np
from minio import Minio
from minio.error import S3Error
from minio.commonconfig import CopySource
import boto3
from botocore.client import Config
from numpy import matrix
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# LDA


############################################################################
# Create a MinIO client object with an endpoint and access/secret keys.
############################################################################
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False,
)

###################################
# SSL Troubleshooting
####################################

# Create a custom SSL context
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE  # Be cautious with this in production

start_time = time.time()


#########################################################################
# Use the minio bucket Text for LDA
#########################################################################
#########################################################################
def fetch_texts_from_minio(bucket_name):
    texts = []
    objects = client.list_objects(bucket_name, recursive=True)
    for obj in objects:
        if obj.object_name.endswith('.csv'):
            response = client.get_object(bucket_name, obj.object_name)
            # Read the CSV and extract text from the first column
            try:
                csv_content = pd.read_csv(BytesIO(response.read()), lineterminator='\n')
                response = client.get_object(bucket_name, obj.object_name)
                metadata = response.get('Metadata', {})
                print("Metadata for the object:", metadata)

                # error_bad_lines=False,  # Skip bad lines
                # warn_bad_lines=True,    # Warn about those lines
                # low_memory=False

                if not csv_content.empty:
                    first_column = csv_content.iloc[:, 0]  # Assume text is in the first column
                    texts.extend(first_column.astype(str).tolist())
            except pd.errors.EmptyDataError:
                print(f"Warning: The file {obj.object_name} is empty and was skipped.")
            except Exception as e:
                print(f"Error processing file {obj.object_name}: {e}")
            finally:
                response.close()
    return texts


####

'''
def keywords_lda(text, numKeywords=10):
    # Function to extract keywords from text using LDA

    # Document-Term Matrix
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([text])

    # LDA
    num_topics = 1  # Number of topics to identify
    lda_model = LatentDirichletAllocation(n_components=num_topics)
    lda_model.fit(dtm)

    # Interpret the topic
    feature_names = vectorizer.get_feature_names_out()
    topic = lda_model.components_[0]
    top_words_indices = topic.argsort()[:-numKeywords - 1:-1]
    top_words = [feature_names[i] for i in top_words_indices]
    top_weights = [topic[i] for i in top_words_indices]

    # Normalize the weights        ------------------------------------------------ Try without normalizing. These weights represent the importance on the overall table, not only with respect to the other keywords  --------------------------------------
    total_weight = sum(top_weights)
    normalized_weights = [weight / total_weight for weight in top_weights]

    # Store the top words and normalized weights in variables
    ### original
    topic_words_and_weights = list(zip(top_words, normalized_weights))
    ###
    topic_words_and_weights = [
        (f"{index + 1}: {word}", weight)
        for index, (word, weight) in enumerate(zip(top_words, normalized_weights))
    ]

    return topic_words_and_weights

'''
####################################################################################
#   MINO ASYNC
####################################################################################

async def download_model(bucket_name, object_name):
    url = client.presigned_get_object(bucket_name, object_name)
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                # Assuming the model is a binary file, adjust if it's saved in other formats
                with open("downloaded_model.bin", "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                print("Model downloaded successfully.")
            else:
                print("Failed to download the model.")


#####################################################################################
# lda_term for the whole bucket
#####################################################################################
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from io import BytesIO


def lda_term_distribution_all_csvs(bucket_name, num_topics=1):
    # List all objects in the bucket that end with .csv
    objects = client.list_objects(bucket_name, recursive=True)
    topic_words = {}

    for obj in objects:
        if obj.object_name.endswith('.csv'):
            try:
                # Fetch CSV file from MinIO
                response = client.get_object(bucket_name, obj.object_name)
                data = pd.read_csv(BytesIO(response.read()))

                # Assume text data is in a specific column, here using the first column for example
                texts = data.iloc[:, 0].astype(str).tolist()

                # Preprocess texts
                texts = [text.lower().strip() for text in texts if isinstance(text, str)]

                # Initialize CountVectorizer with token pattern to exclude numbers
                vectorizer = CountVectorizer(
                    stop_words='english',
                    min_df=2,
                    max_df=0.95,
                    token_pattern=r'(?u)\b[a-zA-Z]+\b'  # Filter all non alphabetic words out
                )

                # Create Document-term matrix
                dtm = vectorizer.fit_transform(texts)

                # Initialize and fit LDA model
                lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0, max_iter=10,
                                                      learning_method='online')
                lda_model.fit(dtm)

                # Extract the words and topic-word distributions, normalize distributions
                words = vectorizer.get_feature_names_out()
                topic_word_distribution = lda_model.components_
                topic_word_distribution /= topic_word_distribution.sum(axis=1)[:, None]  # Normalize

                # Store the words with their respective weights per topic
                file_topics = []
                for topic_dist in topic_word_distribution:
                    topic = {word: weight for word, weight in zip(words, topic_dist)}
                    file_topics.append(topic)

                # Map file name to its topics
                topic_words[obj.object_name] = file_topics

            except Exception as e:
                print(f"Error processing file {obj.object_name}: {e}")

    return topic_words


#########################################################################################
# print of all_csv_topics
#########################################################################################
def print_lda_topics(all_words, all_topic_word_dists):
    for file_idx, (words, topic_word_dists) in enumerate(zip(all_words, all_topic_word_dists)):
        print(f"---------------------------------------------------------------------------")
        print(f"Results for File {file_idx + 1}")
        print(f"---------------------------------------------------------------------------")
        for topic_idx, topic_dist in enumerate(topic_word_dists):
            sorted_word_indices = topic_dist.argsort()[
                                  ::-1]  # Sort indices of the words in the topic by their contribution
            print(f"Topic {topic_idx + 1}:")
            for word_idx in sorted_word_indices[:10]:  # Show top 20 words
                print(f"{words[word_idx]}: {topic_dist[word_idx]:.4f}")
            print(f"---------------------------------------------------------------------------")


#########################################################################################
# for just one file within the bucket


def lda_term_distribution(bucket_name, object_name, num_topics=1):
    # Fetch CSV file from MinIO
    try:
        response = client.get_object(bucket_name, object_name)
        data = pd.read_csv(BytesIO(response.read()))
    except Exception as e:
        print(f"Error fetching or reading file {object_name}: {e}")
        return None

    # Assume text data is in a specific column, here using the first column for example
    texts = data.iloc[:, 0].astype(str).tolist()

    # Enhanced Preprocessing to remove numbers and strip whitespace, convert to lower case
    def preprocess(text):
        # Remove numbers and any non-alphabetic characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lower case and strip whitespace
        return text.lower().strip()

    texts = [preprocess(text) for text in texts]

    # Initialize CountVectorizer with a custom token pattern to ensure only alphabetic words are considered
    vectorizer = CountVectorizer(
        stop_words='english',
        min_df=2,
        max_df=0.95,
        #token_pattern=r'\b[a-zA-Z]+\b'  # only words consisting entirely of letters
        token_pattern = r'\b[a-zA-Z]{2,}\b'
    )

    # Create Document-term matrix
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError:
        print("Empty vocabulary; perhaps the 'min_df' and 'max_df' settings filtered out all words.")
        return None

    # Initialize and fit LDA model
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=0, max_iter=10,
                                          learning_method='online')
    lda_model.fit(dtm)

    # Extract the words and topic-word distributions, normalize distributions
    words = vectorizer.get_feature_names_out()
    topic_word_distribution = lda_model.components_
    topic_word_distribution /= topic_word_distribution.sum(axis=1)[:, None]  # Normalize

    return words, topic_word_distribution

    #####################################################################################
    # Representation of Words and Topic Word Distribution
    #####################################################################################
    # if __name__ == "__main__":
    #bucket_name = "commondatacrawl"
    texts = fetch_texts_from_minio(bucket_name)
    if texts:
        num_topics = 5
        result = lda_term_distribution(texts, num_topics)
        if result:
            words, topic_word_distribution = result
            print("Words and Topic Word Distribution:", words, topic_word_distribution)
        else:
            print("LDA processing failed due to insufficient data.")
    else:
        print("No text data found or error in fetching data.")


#####################################################################################
# Word2Vec
#####################################################################################

def embeddings_word2vec(keywords, weights, model):
    if not isinstance(model, dict):
        raise ValueError("Model must be a dictionary of word vectors.")
    #print('rettung5', keywords)
    embeddings = []
    weights_filtered = []
    i = 0
    for word, weight in zip(keywords, weights):

       # print(f"Processing array {i + 1}/{len(keywords)}")
        i = i + 1
        if word in model:
                embeddings.append(model[word])
                weights_filtered.append(weight)
                #print('rettung7',word, model[word], weight)
        #else:
            #   print(f"Word not found in model: {word}")

    return list(tuple(zip(embeddings, weights_filtered)))


#####################################################################################
# GloVe Adjustments_ Word2Vec
#####################################################################################

def load_glove_model(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# Assuming an existing function that fetches top topic words for each document
# Example function signature: get_topic_words_for_documents()
# This function needs to be implemented according to how you retrieve topics from LDA

# Function to print embeddings for each topic word of each file
def print_topic_embeddings(topic_words, glove_model):
    for file, topics in topic_words.items():
        print(f"File: {file}")
        for idx, words_weights in enumerate(topics):
            print(f" Topic {idx + 1}:")
            for word, weight in words_weights.items():
                if word in glove_model:
                    print(f"  {word}: {glove_model[word]}")
                else:
                    print(f"  {word}: Embedding not found")

########################################################################################
# lda_t
########################################################################################
def preprocess_text(text):
    # Separate concatenated words marked by capital letters
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove non-alphanumeric symbols
    text = re.sub(r'[^\w]', ' ', text)
    # Remove numbers
    text = re.sub(r'[0-9]', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Tokenize text
    # split text into seperate words
    words = text.split()
    #tokens = nltk.word_tokenize(text)
    # Remove stop words
    #tokens = [token for token in tokens if token not in stop_words]
    #split text
    words = text.split()
    # Lemmatize tokens
    #tokens = [lemmatizer.lemmatize(w) for w in tokens]
    # Remove words with length <= 2
    words = [word for word in words if len(word) > 1]
    #tokens = [word for word in tokens if len(word) > 2]
    # Include only verbs, proper nouns, and nouns
    include_features = ['VB', 'NNP', 'NN']
    pos_tagged = nltk.pos_tag(words)
    tokens = [word for word, pos in pos_tagged if pos in include_features]
    # Join tokens into a single string
    return ' '.join(tokens)

def split_text_into_chunks(text, chunk_size=100):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def new_lda_term_distribution_all_csv(bucket_name, csvName, numKeywords):
    # List all objects in the bucket that end with .csv
    all_topic_word_dists = []
    topic_words_and_weights = []
    if csvName.endswith('.csv'):
        try:
            # Fetch CSV file from the bucket
            response = client.get_object(bucket_name, csvName)
            stat = client.stat_object(bucket_name, csvName)
            print("Metadata:", stat.metadata)
            data = pd.read_csv(BytesIO(response.read()))

            # Concatenate all text data from all columns
            all_texts = data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()

            # Preprocess texts: convert to lowercase and strip spaces
            texts = [text.lower().strip() for text in all_texts if isinstance(text, str) and text.strip()]

            if len(texts) == 1:
                texts = split_text_into_chunks(texts[0])

            # Check if there is valid text data
            if len(texts) == 0:
                print("No valid text data found.")
                return topic_words_and_weights
            '''
            ######################################
            # Assume text data is in the first column
            texts = data.iloc[:, 0].astype(str).tolist()

            # Preprocess texts: convert to lowercase and strip spaces
            texts = [text.lower().strip() for text in texts if isinstance(text, str)]
            '''
            # Initialize CountVectorizer
            vectorizer = CountVectorizer(
                stop_words='english',
                min_df=1,
                max_df=0.95,
                token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only include words with at least two alphabetic characters
            )

            # Create a Document-Term Matrix (DTM)
            dtm = vectorizer.fit_transform(texts)

            # Define and fit LDA model
            num_topics = 1  # Identifying one topic since we're extracting k keywords
            lda_model = LatentDirichletAllocation(n_components=num_topics)
            lda_model.fit(dtm)

            # Interpret the topic
            feature_names = vectorizer.get_feature_names_out()
            topic = lda_model.components_[0]
            top_words_indices = topic.argsort()[:-numKeywords - 1:-1]
            top_words = [feature_names[i] for i in top_words_indices]
            top_weights = [topic[i] for i in top_words_indices]

            # Normalize the weights
            total_weight = sum(top_weights)
            normalized_weights = [weight / total_weight for weight in top_weights]
            #print("rettung1", top_words)
            #print("rettung2", normalized_weights)
            # Combine top words with their normalized weights
            topic_words_and_weights = list(zip(top_words, normalized_weights))
            # all_topic_word_dists.append((csvName, topic_words_and_weights))
            #print("rettung1", topic_words_and_weights)
        except Exception as e:
            print(f"Error processing {csvName}: {str(e)}")

    return topic_words_and_weights


#####################################################################################
# Start Calls
#####################################################################################

print("---------------------------------------------------------------------------")
print("                              buckets online                               ")
print("---------------------------------------------------------------------------")

try:
    # Example operation: list all buckets
    buckets = client.list_buckets()
    for bucket in buckets:
        print(f'Bucket: {bucket.name}, Creation Date: {bucket.creation_date}')
except S3Error as e:
    print(f"MinIO S3 error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("---------------------------------------------------------------------------")
print("                     Latent Dirichlent Allocation one word                 ")
print("---------------------------------------------------------------------------")
# Test the function
'''
bucketName = "bagofwordstest"
fileName = "22745259_0_4052170081208609462.csv"
testobject = client.get_object(bucketName, fileName)
file_content = testobject.read()

# keywords_and_weights = keywords_lda(file_content)
### printbefehl. normalerweuise nicht dabei
# for word, weight in keywords_and_weights:
#  print(f"{word}: {weight:.4f}")
'''
print("---------------------------------------------------------------------------")
print("                                LDA current                                ")
print("---------------------------------------------------------------------------")

num_keywords = 10

# words, topic_word_dist = lda_term_distribution("commondatacrawl","CC-MAIN-20150728002301-00000-ip-10-236-191-2.ec2.internal.json.csv", num_topics)
# words, topic_word_dist = lda_term_distribution(file_content, num_topics)


print("---------------------------------------------------------------------------")
print("                                One File                                   ")
print("---------------------------------------------------------------------------")
# Print words with highest probability in each topic
# for topic_idx, topic_dist in enumerate(topic_word_dist):
#  sorted_word_indices = topic_dist.argsort()[::-1]
# print("---------------------------------------------------------------------------")
# print(f"Topic {topic_idx + 1}:")
# for word_idx in sorted_word_indices[:20]:
#   print(f"{words[word_idx]}: {topic_dist[word_idx]:.4f}")

print("---------------------------------------------------------------------------")
print("                                Whole Bucket                               ")
print("---------------------------------------------------------------------------")

#wholebucketname = "trecsmall"
wholebucketname = "commonwebtables"
#wholebucketname = "commonwebcrawlerhuge"
#wholebucketname = "trectables"
#wholebucketname = "commonwebtables"
#wholebucketname = "commondatacrawl2"
#wholebucketname = "commondatacrawl"
#wholebucketname = "bagofwordstest"

# all_words_weigths = lda_term_distribution_all_csvs(wholebucketname, num_topics) for glove shit


print(f"Now calculating on bucket: Bucket: {wholebucketname}")
# print_lda_topics(all_embeddings, all_weights)
glove_model = load_glove_model(
    "/Users/johannesgesk/Documents_MacIntouch/Philipps_Universität_Marburg/2023WS/Bachelor_Arbeit/datasets/GloVe/glove.42B.300d.txt")
# create_embeddings(all_words_weigths,glove_model)
print("---------------------------------------------------------------------------")
print("                                 Word2Vec                                  ")
print("---------------------------------------------------------------------------")


# Print out the types and shapes/values of i[0] and i[1]
# for idx, (emb, weight) in enumerate(embedding_weight_pairs):
#   print(f"Index {idx}: Type of i[0] = {type(emb)}, Content of i[0] = {emb}")
#  print(f"Index {idx}: Type of i[1] = {type(weight)}, Content of i[1] = {weight}")
#   if idx == 10:  # Limit output to the first 10 items for brevity
#      break


def calculateEmbeddings(all_keywords, all_weights, glove_model):
    embedding_weight_pairs = embeddings_word2vec(all_keywords, all_weights, glove_model)
    # check format
    embeddings = np.array([i[0] for i in embedding_weight_pairs], dtype=object)
    new_weights = np.array([i[1] for i in embedding_weight_pairs], dtype=object)
    #print('rettung4', embeddings.shape)
    return np.array(embeddings, dtype=float), np.array(new_weights, dtype=float)


## Out of calculated keywords and embeddings now calculate one representive vector.
## weight each vector with their corresponding weight.
##
def calculateRepresentiveVector(all_keywords, all_weights):
    # Calculate embeddings and weights
    #print('rettung2', all_keywords)

    all_embeddings, embeddings_weights = calculateEmbeddings(all_keywords, all_weights, glove_model)

    # Check if input arrays are empty
    if all_embeddings.size == 0 or embeddings_weights.size == 0:
        print("Input arrays are empty, skipping computation.")
        weighted_embeddings = np.zeros((1, 300))
    else:
        weighted_embeddings = np.average(all_embeddings, axis=0, weights=embeddings_weights)
    # Convert weights into a numpy array for broadcasting if not already
    #if not isinstance(embeddings_weights, np.ndarray):
     #   embeddings_weights = np.array(embeddings_weights, dtype=float)

    # Ensure embeddings are numpy arrays and read y for multiplication
    #if isinstance(all_embeddings[0], list):  # Assuming embeddings are lists, not numpy arrays
     #   all_embeddings = np.array(all_embeddings, dtype=float)

    print("---------------------------------------------------------------------------")
    print("                                 WEIGHTS                                    ")
    print("---------------------------------------------------------------------------")
    # print(embeddings_weights)
    #embeddings_weights = embeddings_weights.reshape(-1, 1)
    # Multiply each embedding by its corresponding weight

    #calculating weighted average
   # weighted_embeddings = np.average(all_embeddings, axis=0, weights=embeddings_weights)
    #print('rettung6', weighted_embeddings.shape)
    #print('rettung8', all_embeddings.shape)
    #print('rettung9', embeddings_weights.shape)

    # print(weighted_embeddings)

    return weighted_embeddings

def calculateRepresentiveVectorForQuery(csvName, all_keywords, all_weights):
    all_representive_vectors_query_ = []
    csvNameStorage = {}
    representiveVector = calculateRepresentiveVector(all_keywords, all_weights)
    #print(f"Querydocument {csvName}: {representiveVector}")
    all_representive_vectors_query_.append(representiveVector)

    # Store all_representive_vectors wihtin a 2D Matrix. Each row is a 300 dimensional vector


    all_representive_vectors_matrix = np.vstack(all_representive_vectors_query_)
    csvNameStorage[0] = csvName
    return all_representive_vectors_matrix, csvNameStorage
def serialize_representative_vector(vector):
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    elif isinstance(vector, tuple):
        return [serialize_representative_vector(v) if isinstance(v, (np.ndarray, tuple)) else v for v in vector]
    else:
        raise ValueError("representative_vector must be either a NumPy array or a tuple")
def update_meta_data_min_io(bucket_name, object_name, existing_metadata, vector_to_add):
    try:

        vector_serialized = serialize_representative_vector(vector_to_add)
        vector_str = json.dumps(vector_serialized)
        #vector_str = json.dumps(vector_to_add)
        # meta data to add
        new_metadata = {
            "X-Amz-Meta-Vector": vector_str
        }
        # Existing user metadata
        #existing_metadata = existing_metadata.metadata

        # Update metadata
        updated_metadata = {**existing_metadata, **new_metadata}
        copy_source = CopySource(bucket_name, object_name)
        # Copy object to the same location with new metadata
        client.copy_object(
            bucket_name,
            object_name,
            copy_source,
            metadata=updated_metadata,
            metadata_directive="REPLACE"
        )

        print(f"Metadata for object '{object_name}' updated successfully.")
    except S3Error as e:
        print(f"Error occurred: {e}")


def initializeLake(bucket_name):
    objects = client.list_objects(bucket_name, recursive=True)
    all_representive_vectors = []
    keywords_list = {}
    csv_name_storage = {}
    i = 0
    for obj in objects:
        # get the Name of the current csv.file
        objName = obj.object_name
        #get the meta data, check if the vector already exists in the metadata
        object_stat = client.stat_object(bucket_name, objName)

        # store the name into an array
        csv_name_storage[i] = objName

        ##############################################################################################
        metadata_key = "X-Amz-Meta-Vector"
        '''
        if metadata_key in object_stat.metadata:
            representiveVector = object_stat.metadata[metadata_key]
            print(f"Metadata key '{metadata_key}' is present.")
            print(f"Metadata value: {representiveVector[-5:]}")
            new_keywords = new_lda_term_distribution_all_csv(bucket_name, objName, num_keywords)
        else:
            # get the top k keywords of the csv file.
            print(f"Metadata key '{metadata_key}' is not present. Calculating now")
            new_keywords = new_lda_term_distribution_all_csv(bucket_name, objName, num_keywords)
            all_keyword = [t[0] for t in new_keywords]
            all_weights = [t[1] for t in new_keywords]
            print("rettung")
            print(objName)
            print(all_keyword)
            print(all_weights)
            representiveVector = calculateRepresentiveVector(all_keyword, all_weights)
            ####################
            update_meta_data_min_io(bucket_name, objName, object_stat.metadata, representiveVector)
        ##############################################################################################
        '''

        # get the top k keywords of the csv file.
        new_keywords = new_lda_term_distribution_all_csv(bucket_name, objName, num_keywords)
        all_keyword = [t[0] for t in new_keywords]
        all_weights = [t[1] for t in new_keywords]
        print("rettung")
        print(objName)
        print(all_keyword)
        print(all_weights)
        representiveVector = calculateRepresentiveVector(all_keyword, all_weights)


        all_representive_vectors.append(representiveVector)
        keywords_list[i] = new_keywords
        i = i + 1
    # Store all_representive_vectors wihtin a 2D Matrix. Each row is a 300 dimensional vector
    all_representive_vectors_matrix = np.vstack(all_representive_vectors)

    return all_representive_vectors_matrix, csv_name_storage, keywords_list


all_calculated_representive_vectors, all_calculated_csv_file_names, all_calculated_keywords_list = initializeLake(wholebucketname)


#####################################################################################################################
##########################################create hypertables
#####################################################################################################################
####################################################################################################################

def nearest_neighbor(queryvektor, all_calculated_representive_vectors1, k=1):
    """
    Input:
      - v, the vector you are going find the nearest neighbor for
      - candidates: a set of vectors where we will find the neighbors
      - k: top k nearest neighbors to find
    Output:
      - k_idx: the indices of the top k closest vectors in sorted form
    """
    ### START CODE HERE (REPLACE INSTANCES OF 'None' with your code) ###
    similarity_l = []

    # for each candidate vector...
    for representativeVector in all_calculated_representive_vectors1:
        representativeVector_adj = representativeVector.reshape(1, -1)
        # get the cosine similarity
        cos_similarity1 = cosine_similarity(queryvektor, representativeVector_adj)
        # append the similarity to the list
        similarity_l.append(cos_similarity1[0][0])

    # sort the similarity list and get the indices of the sorted list
    sorted_ids = np.argsort(similarity_l)

    # get the indices of the k most similar candidate vectors
    k_idx = sorted_ids[-k:]
    ### END CODE HERE ###
    print('Hilfe', k_idx)
    return k_idx


print(print(f"shape of document_vecs {all_calculated_representive_vectors.shape}"))

#########################################################################################################################################
######################################### LSH start#
########################################################################################################################################
# Parameters
N_DIMS = 300  # Dimension of your vectors
num_hyperplanes = 13  # Number of hyperplanes (hash functions)
num_repeat_process = 25

planes_l = [np.random.normal(size=(N_DIMS, num_hyperplanes))
            for _ in range(num_repeat_process)]

print(len(planes_l))  # 25 ways to devide the space
print(len(planes_l[0]))  # 300 dimensional space
print(len(planes_l[0][0]))  # 10 planes in each space


# for the set of planes,
# calculate the dot product between the vector and the matrix containing the planes
# remember that planes has shape (300, 10)
# The dot product will have the shape (1,10)
def hash_value_of_vector(v, planes):
    dot_product = np.dot(v, planes)  # This caluclates the positoin of the vector for each n dimensional line

    # get the sign of the dot product (1,10) shaped vector
    sign_of_dot_product = np.sign(dot_product)

    # set h to be false (eqivalent to 0 when used in operations) if the sign is negative,
    # and true (equivalent to 1) if the sign is positive (1,10) shaped vector
    h = sign_of_dot_product >= 0

    # remove extra un-used dimensions (convert this from a 2D to a 1D array)
    h = np.squeeze(h)

    # initialize the hash value to 0
    hash_value = 0

    n_planes = planes.shape[1]
    for i in range(n_planes):
        # increment the hash value by 2^i * h_i
        hash_value += np.power(2, i) * h[i]

    # cast hash_value as an integer


    hash_value = int(hash_value)
    #print('Hash value of vector', hash_value)
    return hash_value


np.random.seed(0)
idx = 0
planes = planes_l[idx]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
print(f" The hash value for this vector,",
      f"and the set of planes at index {idx},",
      f"is {hash_value_of_vector(vec, planes)}")

print("Done")


# This is the code used to create a hash table: feel free to read over it
def make_hash_table(vecs, planes):
    """
    Input:
        - vecs: list of vectors to be hashed.
        - planes: the matrix of planes in a single "universe", with shape (embedding dimensions, number of planes).
    Output:
        - hash_table: dictionary - keys are hashes, values are lists of vectors (hash buckets)
        - id_table: dictionary - keys are hashes, values are list of vectors id's
                            (it's used to know which tweet corresponds to the hashed vector)
    """

    # number of planes is the number of columns in the planes matrix
    num_of_planes = planes.shape[1]

    # number of buckets is 2^(number of planes)
    num_buckets = 2 ** num_of_planes

    #num_buckets = 100
    # create the hash table as a dictionary.
    # Keys are integers (0,1,2.. number of buckets)
    # Values are empty lists
    hash_table = {i: [] for i in range(num_buckets)}

    # create the id table as a dictionary.
    # Keys are integers (0,1,2... number of buckets)
    # Values are empty lists
    id_table = {i: [] for i in range(num_buckets)}

    # create the vector to hash table as a dictionary.
    # Keys are vector indices
    # Values are the hash values (bucket names)
    vector_to_hash = {}

    # for each vector in 'vecs'
    for i, v in enumerate(vecs):
        # calculate the hash value for the vector
        h = hash_value_of_vector(v, planes)

        # store the vector into hash_table at key h,
        # by appending the vector v to the list at key h
        hash_table[h].append(v)

        # store the vector's index 'i' (each document is given a unique integer 0,1,2...)
        # the key is the h, and the 'i' is appended to the list at key h
        id_table[h].append(i)

        # store the hash value for the vector
        #vector_to_hash[i] = h


    #return hash_table, id_table, vector_to_hash
    return hash_table, id_table

# You do not have to input any code in this cell, but it is relevant to grading, so please do not change anything
'''
np.random.seed(0)
planes = planes_l[0]  # get one 'universe' of planes to test the function
vec = np.random.rand(1, 300)
tmp_hash_table, tmp_id_table = make_hash_table(vec, planes)

print(f"The hash table at key 0 has {len(tmp_hash_table[0])} document vectors")
print(f"The id table at key 0 has {len(tmp_id_table[0])}")
print(f"The first 5 document indices stored at key 0 of are {tmp_id_table[0][0:5]}")
'''
######################################
# create hash table
# Creating the hashtables
hash_tables = []
id_tables = []
for universe_id in range(num_repeat_process):  # there are 25 hashes
    #print('working on hash universe #:', universe_id)
    planes = planes_l[universe_id]
    #hash_table, id_table, bucket_names_table = make_hash_table(all_calculated_representive_vectors, planes)
    hash_table, id_table = make_hash_table(all_calculated_representive_vectors, planes)

    hash_tables.append(hash_table)
    id_tables.append(id_table)

    #############################################################################################
    # R studio for the last universe
    #if universe_id is (num_repeat_process):
        # Initialize lists to store vectors and their corresponding IDs
    all_vectors = []
    all_ids = []
    all_bucketnames= []

    for key in hash_table:
        vectors = hash_table[key]
        ids = id_table[key]
        hash_bucket_names = ids
        #hash_bucket_names = bucket_names_table

        all_vectors.extend(vectors)
        all_ids.extend(ids)
        all_bucketnames.extend([hash_bucket_names] * len(vectors))

    #if universe_id is (num_repeat_process - 1):

        # Convert list of vectors into a DataFrame
    df_vectors = pd.DataFrame(all_vectors)

        # Add IDs as a column to the DataFrame
    df_vectors['hashbucket'] = all_bucketnames

        # Save the DataFrame to a CSV file in the same directory as the script
    project_dir = os.path.dirname(os.path.abspath(__file__))
        #time = time.time()
    csv_file_path = os.path.join(project_dir, f"hash_table_vectors_with_ids_trecsmall.csv")
    df_vectors.to_csv(csv_file_path, index=False)

    print(f"CSV file saved to {csv_file_path}")
        #####################################################################################

print("---------------------------------------------------------------------------")
print("                     LSH                                                   ")
print("---------------------------------------------------------------------------")


#####################################################################################################################
##########################################create hypertables
#####################################################################################################################
# UNQ_C21 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# This is the code used to do the fast nearest neighbor search. Feel free to go over it
# def approximate_knn(doc_id, v, planes_l, k=1, num_universes_to_use=num_repeat_process):
def approximate_knn(csvName, v, planes_l, k=1, num_universes_to_use=num_repeat_process):
    """Search for k-NN using hashes."""
    assert num_universes_to_use <= num_repeat_process

    # Vectors that will be checked as p0ossible nearest neighbor
    vecs_to_consider_l = list()

    # list of document IDs
    ids_to_consider_l = list()

    # create a set for ids to consider, for faster checking if a document ID already exists in the set
    ids_to_consider_set = set()

    # loop through the universes of planes
    for universe_id in range(num_universes_to_use):

        # get the set of planes from the planes_l list, for this particular universe_id
        planes3 = planes_l[universe_id]

        # get the hash value of the vector for this set of planesx
        hash_value = hash_value_of_vector(v, planes3)

        # get the hash table for this particular universe_id
        hash_table = hash_tables[universe_id]

        # get the list of document vectors for this hash table, where the key is the hash_value
        document_vectors_l = hash_table[hash_value]

        # get the id_table for this particular universe_id
        id_table = id_tables[universe_id]

        # get the subset of documents to consider as nearest neighbors from this id_table dictionary
        new_ids_to_consider = id_table[hash_value]




        # remove the id of the document that we're searching
        if csvName in new_ids_to_consider:
            new_ids_to_consider.remove(csvName)
            print(f"removed doc_id {csvName} of input vector from new_ids_to_search")

        # loop through the subset of document vectors to consider
        for i, new_id in enumerate(new_ids_to_consider):

            # if the document ID is not yet in the set ids_to_consider...
            if new_id not in ids_to_consider_set:
                # access document_vectors_l list at index i to get the embedding
                # then append it to the list of vectors to consider as possible nearest neighbors
                document_vector_at_i = document_vectors_l[i]

                # append the new_id (the index for the document) to the list of ids to consider
                vecs_to_consider_l.append(document_vector_at_i)
                ids_to_consider_l.append(new_id)
                # also add the new_id to the set of ids to consider
                # (use this to check if new_id is not already in the IDs to consider)
                ids_to_consider_set.add(new_id)

        ### END CODE HERE ###

    # Now run k-NN on the smaller set of vecs-to-consider.
    print("Fast considering %d vecs" % len(vecs_to_consider_l))

    # convert the vecs to consider set to a list, then to a numpy array
    vecs_to_consider_arr = np.array(vecs_to_consider_l)
    v_array = np.array(v).reshape(1, -1)

    # call nearest neighbors on the reduced list of candidate vectors
    nearest_neighbor_idx_l = nearest_neighbor(v_array, vecs_to_consider_arr, k=k)
    print('Runboy', nearest_neighbor_idx_l)
    print('Runbnoy', ids_to_consider_l)
    # Use the nearest neighbor index list as indices into the ids to consider
    # create a list of nearest neighbors by the document ids
    ids_to_consider_np = np.array(ids_to_consider_l)
    nearest_neighbor_ids = ids_to_consider_np[nearest_neighbor_idx_l]

    #nearest_neighbor_ids = [ids_to_consider_l[idx] for idx in nearest_neighbor_idx_l]

    #nearest_neighbor_ids = [ids_to_consider_l[idx[0]] for idx in nearest_neighbor_idx_l]
    return nearest_neighbor_ids


###############################################################################################
# search document
###############################################################################################

print("---------------------------------------------------------------------------")
print("                      Search Funtion                                       ")
print("---------------------------------------------------------------------------")


def searchQueryDocument(bucketName, csvName, num_keywords):
    queryresult_vector = []

    new_keywords = new_lda_term_distribution_all_csv(bucketName, csvName, num_keywords)
    all_keywords1 = [t[0] for t in new_keywords]
    all_weights1 = [t[1] for t in new_keywords]
    representiveVector, vec_csv_Name = calculateRepresentiveVectorForQuery(csvName, all_keywords1, all_weights1)

    #print(representiveVector)
    queryresult_vector.append(representiveVector)

    representiveVector_matrix = np.vstack(queryresult_vector)

    vec_to_search = representiveVector_matrix[0]

    #print(vec_to_search)

    nearest_neighbor_ids = approximate_knn(vec_csv_Name, vec_to_search, planes_l, k=10, num_universes_to_use=1)

    print(f"Nearest neighbors for document {csvName}")
    print(f"Keywords searched for: {new_keywords}")



    for neighbor_id in nearest_neighbor_ids:
        print(f"Nearest neighbor at document id {all_calculated_csv_file_names[neighbor_id]}")
        print(f"document keywords: {all_calculated_keywords_list[neighbor_id]}")


searchQueryDocument("trecquery", "'Star Wars' Story of Clone Wars Expands in 'Rebels'.csv", 10)

################################################################################################

print("---------------------------------------------------------------------------")
print("                      Word2Vec Glove                                       ")
print("---------------------------------------------------------------------------")

path = ' local_path = ''/Users/johannesgesk/Documents_MacIntouch/Philipps_Universität_Marburg/2023WS/Bachelor Arbeit/datasets/GloVe/glove.42B.300d.txt'

print("---------------------------------------------------------------------------")
print("                                RunTime                                    ")
print("---------------------------------------------------------------------------")

complete_time = time.time() - start_time
print(complete_time)

print("---------------------------------------------------------------------------")
print("                                The End                                    ")
print("---------------------------------------------------------------------------")

'''
#TestTheFunction
bucketName = "modelbucket"
fileName = "glove.42B.300d.txt"

async def download_and_read_model(bucket_name, object_name):
    url = client.presigned_get_object(bucket_name, object_name)
    model = {}

    # Setting up an HTTP session
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                print(f"Response status: {response.status} OK")
                # Reading the response content as text
                text = await response.text()
                lines = text.strip().split('\n')
                for line in lines:
                    parts = line.split()
                    if len(parts) > 1:
                        word = parts[0]
                        try:
                            # Ensure that vector parts can be converted to floats
                            vector = list(map(float, parts[1:]))
                            model[word] = vector
                        except ValueError:
                            print(f"Skipping line due to error converting to float: {line}")
                            continue
            else:
                print(f"Failed to download the model, status code: {response.status}")
                return None
    return model


#async def main():
    local_path = '/Users/johannesgesk/Documents_MacIntouch/Philipps_Universität_Marburg/2023WS/Bachelor Arbeit/datasets/GloVe/glove.42B.300d.txt'
    model_data = None

    # Check if the model exists locally
    if os.path.exists(local_path):
        # Load local model data
        print("Local use")
        with open(local_path, 'r', encoding='utf-8') as file:
            model_data = file.read()
            print(model_data[:500])  # Print the first 500 characters of the model data
            #################

            ###########
    else:
        print("Download started")
        # Download and read model data if not available locally
        bucketName = "modelbucket"
        fileName = "glove.42B.300d.txt"
        model_data = await download_and_read_model(bucketName, fileName)
        if model_data:
            print(model_data[:500])  # Print the first 500 characters of the model data
        else:
            print("Failed to retrieve the model.")

    if model_data:
        print("Now working on vector calculation")
        #keyword_embeddings = embeddings_word2vec(all_words, all_topic_word_dists, model_data)
        #print(keyword_embeddings)
        test_words = ['king', 'queen', 'man', 'woman']
        for word in test_words:
            if word in model_data:
                print(f"Vector for '{word}':", model_data[word][:5])  # Show first 5 elements of vector
            else:
                print(f"Word '{word}' not found in model.")
asyncio.run(main())

'''
