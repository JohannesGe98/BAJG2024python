import string
import time
import re

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from io import BytesIO
from minio import Minio
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

client = Minio(
    "127.0.0.1:9001",
    access_key="d5dwxTDYc6YBZzgEGA5t",
    secret_key="W7nMjJeIFTbPpQH7ylGmiurHFe1pNFHQEo5X8Dsa",
    secure=False,
)

def load_glove_model(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

model_Word2Vec = load_glove_model(
    "/Users/johannesgesk/Documents_MacIntouch/Philipps_Universität_Marburg/2023WS/Bachelor_Arbeit/datasets/GloVe/glove.42B.300d.txt")


def preprocess_text(x):
    x = re.sub(r'([a-z])([A-Z])', r'\1 \2', x) # Separate concatenated words marked by cappital letters
    x = x.lower() # text to lowercase
    x = re.sub(r'http\S+', '',x) # remove URLs
    x = re.sub(r'[^\w]', ' ', x) # remove not alphanumeric symbols
    x = re.sub(r'[0-9]', '',x) # remove numbers
    x = x.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # remove punctuation
    x = nltk.word_tokenize(x) # tokenize
    x = [token for token in x if token not in stop_words] # remove stop words
    #x = [lemmatizer.lemmatize(w) for w in x] # lemmatization
    x = [word for word in x if len(word) > 2] # remove single characters
    include_features = ['VB', 'NNP', 'NN'] # verbs, proper nouns and nouns (we think this is what makes the difference between topics. Adjectives, for instance, would be interesting for sentiment analysis of these topics, but not for topic modeling)
    pos_tagged = nltk.pos_tag(x) # tag words
    x = [text for text, pos in pos_tagged if pos in include_features] # select only the words tagged as the type we are interested in
    x = ' '.join(x) # join all lemmatized words in one single text variable
    return x

stop_words = stopwords.words('english') # Define stop words
new_sw = ['none', 'nan', 'unnamed']
stop_words.extend(new_sw) # Extend stop words list with custom words

def keywords_lda(text, numKeywords=10):
    topic_words_and_weights = []
    try:
        if len(text) == 0:
            print("No valid text data found.")
            return topic_words_and_weights
        text = preprocess_text(text)
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
        # Handle cases where fewer than numKeywords are found
        if len(top_words) < numKeywords:
            missing_count = numKeywords - len(top_words)
            top_words.extend([''] * missing_count)
            top_weights.extend([0] * missing_count)
        print(top_words)

        # Normalize the weights
        total_weight = sum(top_weights)
        normalized_weights = [weight / total_weight for weight in top_weights]

        # Store the top words and normalized weights in variables
        topic_words_and_weights = list(zip(top_words, normalized_weights))
    except Exception as e:
        print(f"Error processing: {str(e)}")
    return topic_words_and_weights

def embeddings_word2vec(keywords, weights, model):
    embeddings = []
    weights_filtered = []
    for word, weight in zip(keywords, weights):
        if word in model:
            embeddings.append(model[word])
            weights_filtered.append(weight)
    return list(tuple(zip(embeddings, weights_filtered)))

def find_keywords(text, method):
    if method == "lda":
        keywords = keywords_lda(text)
    elif method == "yake":
        print("Fehler")
    else:
        raise ValueError(f"The method to extract the keywords has to be either 'lda' or 'yake'. '{method}' is not accepted")
    return keywords

def find_embeddings(texts, methodKeywords, methodEmbeddings, model, tokenizer=None):
    all_embeddings = []
    all_weights = []
    text = ' '.join(texts)

    keywords_weights = find_keywords(text, methodKeywords)
    keywords = [t[0] for t in keywords_weights]
    weights = [t[1] for t in keywords_weights]
    if methodEmbeddings == "word2vec":
        embeddings_weights = embeddings_word2vec(keywords, weights, model)
        embeddings = np.array([i[0] for i in embeddings_weights])
        weights = np.array([i[1] for i in embeddings_weights])
    elif methodEmbeddings == "bert":
        print("Fehler")
    else:
        raise ValueError(f"The method to extract the keywords has to be either 'word2vec' or 'bert'. '{methodEmbeddings}' is not accepted")
    '''if embeddings.size > 0:
            all_embeddings.append(embeddings)
            all_weights.append(new_weights)
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        final_weights = np.hstack(all_weights)
    else:
        final_embeddings = np.array([])
        final_weights = np.array([])
    '''
    return embeddings, weights

def weighted_centroid_cosine_similarity(embeddings1, weights1, embeddings2, weights2):
    weighted_centroid1 = np.average(embeddings1, axis=0, weights=weights1)
    weighted_centroid2 = np.average(embeddings2, axis=0, weights=weights2)
    cosine_sim = cosine_similarity(weighted_centroid1.reshape(1, -1), weighted_centroid2.reshape(1, -1))
    return cosine_sim[0, 0]


def weighted_pairwise_cosine_similarity(embeddings1, weights1, embeddings2, weights2, i, aggregation='mean'):
    pairwise_similarities = cosine_similarity(embeddings1, embeddings2)
    weighted_pairwise_similarities = pairwise_similarities * np.outer(weights1, weights2)
    if aggregation == 'mean':
        similarity = np.mean(weighted_pairwise_similarities)
    elif aggregation == 'max':
        similarity = np.max(weighted_pairwise_similarities)
    elif aggregation == 'sum':
        similarity = np.sum(weighted_pairwise_similarities)
    return similarity




def rank_embeddings(input_table_text, data_lake, keywordExtractionMethod, embeddingsMethod, model, tokenizer=None, k=5):

    bucket_name_data_lake = data_lake
    table_text = client.list_objects(bucket_name_data_lake, recursive=True)
    numTables = sum(1 for _ in table_text)
    bucket_name_query = "querycommonwebtableshuge"
    response = client.get_object(bucket_name_query, input_table_text)
    data = pd.read_csv(BytesIO(response.read()))
    all_texts = data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()
    texts_query = [text.lower().strip() for text in all_texts if isinstance(text, str) and text.strip()]
    embeddings_input, weights_input = find_embeddings(texts_query, keywordExtractionMethod, embeddingsMethod, model, tokenizer)



    similarities = np.zeros(numTables)
    table_names = []
    table_text = client.list_objects(bucket_name_data_lake, recursive=True)
    for i, obj in enumerate(table_text):
        objName = obj.object_name
        table_names.append(objName)
        response = client.get_object(bucket_name_data_lake, objName)
        data = pd.read_csv(BytesIO(response.read()))
        all_texts = data.apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).tolist()
        current_table_text = [text.lower().strip() for text in all_texts if isinstance(text, str) and text.strip()]
        embeddings_table, weights_table = find_embeddings(current_table_text, keywordExtractionMethod, embeddingsMethod, model, tokenizer)
        if np.size(embeddings_table) > 0:
            similarities[i] = weighted_pairwise_cosine_similarity(embeddings_input, weights_input, embeddings_table, weights_table, i)
            print(similarities[i], objName)
        else:
            similarities[i] = 0.0


    if k is None:
        k = numTables
    elif k > numTables:
        print(f"The introduced k (k = {k}) is larger than the number of tables in the data lake. Thus, the output is the ranking of all tables in the data lake.")
        k = numTables
    ranked_indices = np.argsort(similarities)[::-1][:k]
    ranked_tables_filtered = [(table_names[i], similarities[i]) for i in ranked_indices]

    return ranked_tables_filtered

input_table_text = "1946FootballTeam--UniversityofMichiganAthletics.csv"
keywordExtractionMethod = 'lda'
embeddingsMethod = 'word2vec'
data_lake = "querycommonwebtableshuge"

######################################################################################################################################################
                                                        #call for similarity
######################################################################################################################################################

start_time = time.time()
rank_lda_word2vec_filtered = rank_embeddings(input_table_text, data_lake, keywordExtractionMethod, embeddingsMethod, model_Word2Vec, k=30)

for table, similarity in rank_lda_word2vec_filtered:
    print(f"Table: {table}, Similarity: {similarity}")
    running_time = time.time() - start_time
print(f"runtime: {running_time}")

