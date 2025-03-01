import ollama
from typing import List
from gensim.corpora import Dictionary 
import pandas as pd
# import argparse 

#TODO: maybe integrate a few-shots approach by passing some examples of human-made TM


def label_topics(topics : List[List[str]]):
    pass


def batched_corpus(c : List[str], batch_size = 100):
    
    b = []
    for doc in c:
        b.append(doc)
        if len(b) >= batch_size:
            yield b

    if b:
        yield b


def get_batch_topics(doc : List[str], client : ollama.Client, model_name : str) -> List[str]:

    prompt = f"""{doc}"""
    topics = client.generate(model=model_name, prompt=prompt)
    return topics


def merge_topics(topics : List[str], client : ollama.Client, model_name : str):

    topics = set(topics)
    prompt = ""
    merged_topics = client.generate(model=model_name, prompt=prompt)
    return merged_topics


def get_corpus_topics(corpus : List[str], client : ollama.Client,  model_name : str):
    

    corpus_topics = []
    for batch in batched_corpus(corpus):
        topics = get_batch_topics(batch, client, model_name)
        corpus_topics.extend(topics)

    return merge_topics(corpus_topics)
 

def main():

    MODEL_NAME = "mistral:7b-instruct-q4_K_M"
    if not MODEL_NAME in ollama.list():
        ollama.pull(MODEL_NAME)
    client = ollama.Client()

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:500]
    texts_list = df["text"].tolist()

    topics = get_corpus_topics(texts_list)
    print(f"Retrived topics: {topics}")

    
