import ollama
from typing import List
import pandas as pd
from pathlib import Path
import re
import pickle


# def batched_corpus(c : List[str], batch_size = 128):
#     b = []
#     for doc in c:
#         b.append(doc)
#         if len(b) >= batch_size:
#             yield "\n\n///\n\n".join(b)
#             b = []

#     if b:
#         yield "///".join(b)

def get_prompt_content(ref : str):
    if not ref.endswith(".txt"):
        content = ref
    else:
        with open(ref, "r", encoding="utf-8") as f:
            content = f.read()
    return content

def cleaned_response(content):
    pattern = r"\d+\.\s\*\*(.*?)\*\*"
    topics = re.findall(pattern, string=content)

    return topics


def get_corpus_topics(corpus : List[str], settings : dict, client : ollama.Client,  model_name : str):
    
    sys_prompt = get_prompt_content(settings["system_prompt"]["content"])
    base_prompt = get_prompt_content(settings["base_prompt"]["content"])
    fs = get_prompt_content(settings["base_prompt"]["fs"])
    st = settings["base_prompt"]["seed_topics"]
    opt = {"temperature" : settings["temp"]}

    # client.chat(model=model_name, messages=[{"role" : "system", "content" : sys_prompt}], options=opt)

    corpus_topics = []
    for doc in corpus:
        # print(batch)
        interpolated_prompt = base_prompt.format(SEED_TOPICS=st, FEW_SHOTS=fs, BATCH=doc)
        response = client.generate(model=model_name, prompt=interpolated_prompt, options=opt)["response"]
        print(f"\n\n{'-'*10}START OF RESPONSE{'-'*10}")
        print(response)
        print(f"\n{'-'*10}END OF RESPONSE{'-'*10}\n")
        topics = cleaned_response(response)
        print(topics)
        corpus_topics.extend(topics)

    merge_prompt = get_prompt_content(settings["merge_prompt"]["content"])
    return merge_topics(corpus_topics, prompt=merge_prompt, client=client, model_name=model_name, options=opt)
 

def merge_topics(topics : List[str], prompt : str | Path, client : ollama.Client, model_name : str, options):
    
    topics = set(topics)
    merged_topics = client.generate(model=model_name, prompt=prompt, options=options)
    return merged_topics


def main():
    # mistral:7b-instruct-q4_K_M
    # deepseek-r1:8b
    MODEL_NAME = "deepseek-r1:8b"
    settings = {
        "system_prompt" : {
            "content" : "prompts/tm_prompt.txt"
        },
        "base_prompt":{
            "content" : "prompts/tm_prompt.txt",
            "seed_topics" : ["International Relations", "War", "Peace", "Cooperation", "Countries"],
            "fs" : "prompts/few_shots.txt"
        },
        "merge_prompt":{
            "content" : "prompt/topic_merge.txt"
        },
        "temp" : 0.0
    }
    models_list = [info.model for info in ollama.list()["models"]]
    if not MODEL_NAME in models_list:
        ollama.pull(MODEL_NAME)

    client = ollama.Client()

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:500]
    texts_list = df["text"].tolist()


    topics = get_corpus_topics(texts_list, settings=settings, client=client, model_name=MODEL_NAME)
    print(f"Retrived topics: {topics}")

if __name__ == "__main__":
    main()
