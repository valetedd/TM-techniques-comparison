import ollama
from typing import List
from gensim.corpora import Dictionary 
import pandas as pd
from pathlib import Path
from functools import partial


def batched_corpus(c : List[str], batch_size = 128):
    b = []
    for doc in c:
        b.append(doc)
        if len(b) >= batch_size:
            yield "///".join(b)

    if b:
        yield "///".join(b)

def get_prompt_content(ref : str):
    if not ref.endswith(".txt"):
        content = ref
    else:
        with open(ref, "r", encoding="utf-8") as f:
            content = f.read()
    return content



def get_corpus_topics(corpus : List[str], settings : dict, client : ollama.Client,  model_name : str):
    
    prompt_content = settings["base_prompt"]["content"]
    prompt = get_prompt_content(prompt_content)
    
    fs_content = settings["base_prompt"]["fs"]
    fs = get_prompt_content(fs_content)

    st = settings["base_prompt"]["seed_topics"]

    corpus_topics = []
    for batch in batched_corpus(corpus):
        interpolated_prompt = prompt.format(SEED_TOPICS=st, FEW_SHOTS=fs, BATCH=batch)
        topics = client.generate(model=model_name, prompt=interpolated_prompt)["response"]
        print(topics)
        corpus_topics.extend(topics)

    merge_prompt = get_prompt_content(settings["merge_prompt"]["content"])
    return merge_topics(corpus_topics, prompt=merge_prompt, client=client, model_name=model_name)
 

def merge_topics(topics : List[str], prompt : str | Path, client : ollama.Client, model_name : str):

    topics = set(topics)
    merged_topics = client.generate(model=model_name, prompt=prompt)
    return merged_topics


def main():

    MODEL_NAME = "mistral:7b-instruct-q4_K_M"

    if not MODEL_NAME in ollama.list():
        ollama.pull(MODEL_NAME)

    client = ollama.Client()

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:500]
    texts_list = df["text"].tolist()

    settings = {
        "base_prompt":{
            "content" : "prompts/tm_prompt.txt",
            "seed_topics" : ["International Relations", "War", "Peace", "Cooperation", "Countries"],
            "fs" : "prompts/few_shots.txt"
        },
        "merge_prompt":{
            "content" : "prompt/topic_merge.txt"
        }
    }

    topics = get_corpus_topics(texts_list, settings=settings, client=client, model_name=MODEL_NAME)
    print(f"Retrived topics: {topics}")

if __name__ == "__main__":
    main()
