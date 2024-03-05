import re
import os
import glob
import boto3
import jsonify
import unidecode
import numpy as np
import pandas as pd

from openai import OpenAI
from io import BytesIO, StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

s3 = boto3.client('s3')

embedding_engine = 'text-embedding-ada-002'
OPENAI_API_KEY = "OPENAI_API_KEY"

client = OpenAI(
    api_key = OPENAI_API_KEY
)

@retry(wait=wait_random_exponential(min=15, max=60), stop=stop_after_attempt(10))
def embedding_with_backoff(**kwargs):
    return client.embeddings.create(**kwargs)

def get_openai_embedding(text, model=embedding_engine):
    result = embedding_with_backoff(
        model=model,
        input=[text]
    )
    return result.data[0].embedding

def compute_doc_embeddings_openai(df: pd.DataFrame):
    return {idx: get_openai_embedding(r.content) for idx, r in df.iterrows()}

def get_text_from_md_file(s3, bucket, key):
    s3Object = s3.get_object(Bucket=bucket, Key=key)
    file_buffer = BytesIO(s3Object['Body'].read())
    
    decoded_content = unidecode.unidecode(content)

    return decoded_content

def get_document_embeddings(text):
    chunk_size = 1500
    chunk_overlap = 150

    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[" \n---\n  ","\n\n", "\n", "(?<=\-.)", " ", ""]
    )

    splits = r_splitter.split_text(text)

    df_text = pd.DataFrame(splits, columns=['content'])

    openai_embeddings = pd.DataFrame(compute_doc_embeddings_openai(df_text))
    
    return df_text, openai_embeddings

def upload_documents(raw_text, pages, df_text, df_embeddings, s3, bucket, key):
    csv_buffer = StringIO()
    csv_vector_buffer = StringIO()
    
    document_name_array = os.path.splitext(key)
    
    txt_document_name = document_name_array[0] + '.txt'
    s3.put_object(Body=raw_text, Bucket=bucket, Key=txt_document_name)
    
    # Upload file with the text for embeddings
    df_text.to_csv(csv_buffer, index=False)
    csv_filename = document_name_array[0] + '_embeddings_text.csv'
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=csv_filename)

    # Upload file with the embeddings
    df_embeddings.to_csv(csv_vector_buffer, index=False)
    csv_embeddings_filename = document_name_array[0] + '_embeddings.csv'
    s3.put_object(Body=csv_vector_buffer.getvalue(), Bucket=bucket, Key=csv_embeddings_filename)

    return txt_document_name, txt_document_name_pages, csv_filename, csv_embeddings_filename

def lambda_handler(event, context):
    try:
        s3 = boto3.client('s3')
        region = "us-east-1"

        bucket = event["bucket"]
        upload_bucket = event["upload_bucket"]
        key = event["key"]
        
        raw_text = get_text_from_md_file(s3, bucket, key)

        df_text, df_embeddings = get_document_embeddings(raw_text)

        txt_document_name, csv_filename, csv_embeddings_filename = upload_documents(raw_text, df_text, df_embeddings, s3, upload_bucket, key)

        payload = {"txt_document_name": txt_document_name,
                "embeddings_text_filename": csv_filename,
                "embeddings_filename": csv_embeddings_filename}

        return jsonify({"message": payload}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500