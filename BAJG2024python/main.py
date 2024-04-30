from minio import Minio
from minio.error import S3Error
import ssl
import logging
#logging.basicConfig(level=logging.DEBUG)

import pandas as pd
from io import BytesIO

####################################
### import Masterarbeit #######

# LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Word embeddings

####################################

# Create a custom SSL context
context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE  # Be cautious with this in production

# Create a MinIO client object with an endpoint and access/secret keys.
client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False,
)




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
                csv_content = pd.read_csv(BytesIO(response.read()))
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


def keywords_lda(text, numKeywords = 10):
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
        (f"{index+1}: {word}", weight)
        for index, (word, weight) in enumerate(zip(top_words, normalized_weights))
    ]


    return topic_words_and_weights

def vector_LDA(text):
    num_topics=1
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=num_topics)
    lda_model.fit(dtm)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#####################################################################################
#lda_term for the whole bucket
def lda_term_distribution_all_csvs(bucket_name, num_topics=1):
    # List all objects in the bucket that end with .csv
    objects = client.list_objects(bucket_name, recursive=True)
    all_words = []
    all_topic_word_dists = []

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

                # Initialize CountVectorizer
                vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.95)

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

                # Collect results from all files
                all_words.append(words)
                all_topic_word_dists.append(topic_word_distribution)

            except Exception as e:
                print(f"Error processing file {obj.object_name}: {e}")

    return all_words, all_topic_word_dists
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

    # Preprocess texts
    texts = [text.lower().strip() for text in texts if isinstance(text, str)]

    # Initialize CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', min_df=2, max_df=0.95)

    # Create Document-term matrix
    try:
        dtm = vectorizer.fit_transform(texts)
    except ValueError:
        print("Empty vocabulary; check the 'min_df' and 'max_df' settings.")
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
    #Representation of Words and Topic Word Distribution
#####################################################################################
#if __name__ == "__main__":
    bucket_name = "commondatacrawl"
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


print("---------------------------------------------------------------------------")
print("                                    buckets                                ")
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
print("                     Latent Dirichlent Allocation                          ")
print("---------------------------------------------------------------------------")
# Test the function
bucketName = "bagofwordstest"
fileName = "22745259_0_4052170081208609462.csv"
testobject = client.get_object(bucketName, fileName)
file_content = testobject.read()

keywords_and_weights = keywords_lda(file_content)
### printbefehl. normalerweuise nicht dabei
for word, weight in keywords_and_weights:
   print(f"{word}: {weight:.4f}")

print("---------------------------------------------------------------------------")
print("                                LDA current                                ")
print("---------------------------------------------------------------------------")


num_topics =10

words, topic_word_dist = lda_term_distribution("commondatacrawl","CC-MAIN-20150728002301-00000-ip-10-236-191-2.ec2.internal.json.csv", num_topics)
#words, topic_word_dist = lda_term_distribution(file_content, num_topics)


print("---------------------------------------------------------------------------")
print("                                One File                                   ")
print("---------------------------------------------------------------------------")
# Print words with highest probability in each topic
for topic_idx, topic_dist in enumerate(topic_word_dist):
    sorted_word_indices = topic_dist.argsort()[::-1]
    print("---------------------------------------------------------------------------")
    print(f"Topic {topic_idx + 1}:")
    for word_idx in sorted_word_indices[:20]:
        print(f"{words[word_idx]}: {topic_dist[word_idx]:.4f}")

print("---------------------------------------------------------------------------")
print("                                Whole Bucket                               ")
print("---------------------------------------------------------------------------")

all_words, all_topic_word_dists= lda_term_distribution_all_csvs("commondatacrawl", num_topics)
def print_lda_topics(all_words, all_topic_word_dists):
    for file_idx, (words, topic_word_dists) in enumerate(zip(all_words, all_topic_word_dists)):
        print(f"---------------------------------------------------------------------------")
        print(f"Results for File {file_idx + 1}")
        print(f"---------------------------------------------------------------------------")
        for topic_idx, topic_dist in enumerate(topic_word_dists):
            sorted_word_indices = topic_dist.argsort()[::-1]  # Sort indices of the words in the topic by their contribution
            print(f"Topic {topic_idx + 1}:")
            for word_idx in sorted_word_indices[:20]:  # Show top 20 words
                print(f"{words[word_idx]}: {topic_dist[word_idx]:.4f}")
            print(f"---------------------------------------------------------------------------")
print_lda_topics(all_words, all_topic_word_dists)
print("---------------------------------------------------------------------------")
print("                                The End                                    ")
print("---------------------------------------------------------------------------")