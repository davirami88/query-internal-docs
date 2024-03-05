import re
import os
import json
import boto3
import jsonify
import unidecode
import numpy as np
import pandas as pd

from openai import OpenAI
from io import BytesIO, StringIO
from sklearn.metrics.pairwise import cosine_similarity

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

s3 = boto3.client("s3")
s3_text = boto3.session.Session().resource('s3')

embedding_engine = "text-embedding-ada-002"
GPT3_turbo = "gpt-3.5-turbo-1106"

OPENAI_API_KEY = "OPENAI_API_KEY"

client = OpenAI(
    api_key = OPENAI_API_KEY
)

@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(10))
def embedding_with_backoff(**kwargs):
    return client.embeddings.create(**kwargs)

@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(10))
def chat_completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs, timeout=30)

def get_openai_embedding(text, model=embedding_engine):
    result = embedding_with_backoff(
        model=model,
        input=[text]
    )
    return result.data[0].embedding

def vector_similarity(x, y):
    return cosine_similarity(np.array(x).reshape(1, -1), np.array(y).reshape(1, -1))[0][0]

def compute_doc_embeddings_openai(df: pd.DataFrame):
    return {idx: get_openai_embedding(r.content) for idx, r in df.iterrows()}

def order_document_sections_by_query_similarity_openai(query, contexts):
    query_embedding = get_openai_embedding(query)
    document_similarities = sorted([(vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()], reverse=True)
    return document_similarities

def get_openai_response(prompt, text, model):
    response = chat_completion_with_backoff(model=model,
                                            response_format={ "type": "json_object" },
                                            messages=[{"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                                                      {"role": "user", "content": text + prompt}],
                                            temperature=0)
    try:
        result = json.loads(response.choices[0].message.content)
    except:
        result = []

    return result

def get_query_response(s3_bucket_name, document_name, key_csv_text_file, key_csv_embeddings_file, query):
    try:
        # Get the files from the S3 object
        res = s3_text.Object(s3_bucket_name, document_name)

        file_content = res.get()['Body'].read().decode('utf-8')
        json_content = json.loads(file_content)

        text = []
        for page in json_content:
            text.append(page[0])

        text = ' '.join(text)
        
        obj_text = s3.get_object(Bucket=s3_bucket_name, Key=key_csv_text_file)
        obj_embeddings = s3.get_object(Bucket=s3_bucket_name, Key=key_csv_embeddings_file)

        # Read the object's content into pandas DataFrame
        df_text = pd.read_csv(BytesIO(obj_text['Body'].read()))
        df_embeddings = pd.read_csv(BytesIO(obj_embeddings['Body'].read()))
        
        text_ranking = order_document_sections_by_query_similarity_openai(query, df_embeddings)[:25]

        separator = "\n\n------\n\n"  # Clear visual separator
        prompt_check_query = """Based on the text provided, you are to determine if it is possible to answer this question: """ + query + """
        Please answer "yes" only if the answer to the question is in the text explicitly, clearly and unambiguously, and if the text contains the specific information to answer the question.
        If is not possible to answer based on the text or does not directly and clearly identify the answer, or if there is any ambiguity or assumption required to identify them, you must answer "no".
        Return your answer on the key "answer".
        """

        prompt = """Based on the text, which is a selected part of a whole documentation, return a JSON with a single answer in the key "answer". 
        You must not recommend read the docs, beacause the text is an specific part of the whole documentation. 
        Try to be as generic as possible and always based your answer on the text provided. Question: """ + query 

        valid_text_options = []

        for option in text_ranking:
            text_index = int(re.findall(r'\d+', str(option[1]))[0])
            possible_text = text[text.find(df_text.loc[text_index, "content"][:30]) - 500:text.find(df_text.loc[text_index, "content"][:30]) + 2500]

            response = get_openai_response(prompt_check_query, possible_text + separator, GPT3_turbo)

            if response['answer'] == 'yes':
                valid_text_options.append(possible_text)

        if len(valid_text_options) > 0:
            valid_text = " ".join(valid_text_options)
            
            response_user_query = get_openai_response(prompt, valid_text + separator, GPT3_turbo)

            return response_user_query['answer']
        else:
            return "openai_error"
    except:
        return "s3_download_error"

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    region = "us-east-1"

    try:
        bucket = event["origin_bucket"]
        key_txt_pages = event['txt_pages']
        key_csv_text_file = event['csv_text_file']
        key_csv_embeddings_file = event['csv_embeddings_file']
        query = event['query']

        response = get_query_response(bucket, key_txt_pages, key_csv_text_file, key_csv_embeddings_file, query)

        payload = {"payload": response }

        return jsonify({"message": payload}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500