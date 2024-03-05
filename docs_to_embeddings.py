import re
import os
import glob
import boto3
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

#Future function to generate the embeddings from an S3 file
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
    
#Future function to upload the embeddings to S3
def upload_documents(raw_text, pages, df_text, df_embeddings, s3, bucket, key):
    csv_buffer = StringIO()
    csv_vector_buffer = StringIO()
    
    document_name_array = os.path.splitext(key)
    
    # Upload both .txt files
    txt_document_name = document_name_array[0] + '.txt'
    s3.put_object(Body=raw_text,Bucket=bucket,Key=txt_document_name)

    # Upload file with the text for embeddings
    df_text.to_csv(csv_buffer, index=False)
    csv_filename = document_name_array[0] + '_embeddings_text.csv'
    s3.put_object(Body=csv_buffer.getvalue(), Bucket=S3_BUCKET, Key=csv_filename)

    # Upload file with the embeddings
    df_embeddings.to_csv(csv_vector_buffer, index=False)
    csv_embeddings_filename = document_name_array[0] + '_embeddings.csv'
    s3.put_object(Body=csv_vector_buffer.getvalue(), Bucket=S3_BUCKET, Key=csv_embeddings_filename)

    n_of_pages = len(pages)

    return txt_document_name, n_of_pages

def lambda_handler(event, context):
    s3 = boto3.client('s3')
    region = "us-east-1"

    #TODO the code to read the "event" to get the parameters and run the functions above