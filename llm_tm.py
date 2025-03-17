import ollama
from typing import List
import pandas as pd
from pathlib import Path
import re
import pickle
import requests

# def batched_corpus(c : List[str], batch_size = 128):
#     b = []
#     for doc in c:
#         b.append(doc)
#         if len(b) >= batch_size:
#             yield "\n\n///\n\n".join(b)
#             b = []

#     if b:
#         yield "///".join(b)


class LLM_TopicModel():
    def __init__(self, settings, model, sys : bool = False):

        self.status = self._check_ollama_reqs()
        self.client = ollama.Client()
        self.model = model

        if sys:
            sys_p = settings["system_prompt"]["content"]
            self.sys_prompt = self._get_prompt(sys_p)

        base_p = settings["base_prompt"]["content"]
        self.base_prompt = self._get_prompt(base_p)

        few_shots = settings["base_prompt"]["fs"]
        self.fs = self._get_prompt(few_shots)
        
        self.merge_prompt = self._get_prompt(self.settings["merge_prompt"]["content"])

        self.st = settings["base_prompt"]["seed_topics"]
        self.opt = settings["options"]
        
    
    def _check_ollama_reqs(self, model_name):
        try:
            models_list = [info.model for info in ollama.list()["models"]]
            if not model_name in models_list:
                ollama.pull(model_name)
                print(f"Model {model_name} not found. Please pull it with 'ollama pull {model_name}'")
                return False
            return True
        except requests.exceptions.ConnectionError:
            print("Ollama is not running. Please start Ollama service.")
            return False


    def _get_prompt(self, ref : str):
        
        if not isinstance(ref, str):
            raise TypeError("'ref' must either be a string-type prompt or a path to a txt")
        
        if not ref.endswith(".txt"):
            content = ref
        else:
            with open(ref, "r", encoding="utf-8") as f:
                content = f.read()
        return content
    

    def _extract_topic(self, content):
        pattern = r"(?:\d+\.|\-)\s\*\*(.*?)\*\*"
        topics = re.findall(pattern, string=content)

        return topics


    def get_topics(self, corpus : List[str]):

        corpus_topics = []
        for doc in corpus:
            # print(batch)
            interpolated_prompt = self.base_prompt.format(SEED_TOPICS=self.st, FEW_SHOTS=self.fs, BATCH=doc)
            response = self.client.generate(model=self.model_name, prompt=interpolated_prompt, options=self.opt)["response"]

            print(f"\n\n{'-'*10}START OF RESPONSE{'-'*10}")
            print(response)
            print(f"\n{'-'*10}END OF RESPONSE{'-'*10}\n")

            topics = self._extract_topic(response)
            print(topics)
            corpus_topics.extend(topics)
        self.topics = self.merge_topics(corpus_topics, settings=self.settings, client=self.client, model_name=self.model_name)
        return self.topics
    

    def merge_topics( self, topics : List[str]):


        topics = list(set(topics)) # dedupe
        merged_topics = self.client.generate(model=self.model_name, prompt=self.merge_prompt, options=self.opt)
        return merged_topics


    def _extract_label(self, content : str):
        return content

    def get_topic_labels(self, topics : List[List[str]]):
        
        base_prompt = self._get_prompt(self.settings["base_prompt"]["content"])

        label2topic = {}
        for topic_list in topics:
            resp = self.client.generate(model=self.model, prompt=base_prompt, options=self.opt)["response"]
            label = self._extract_label(resp)
            label2topic[label] = topic_list

        return label2topic


def main():
    # mistral:7b-instruct-q4_K_M
    # deepseek-r1:8b
    MODEL_NAME = "deepseek-r1:8b"

    settings = {
        "system_prompt" : {
            "content" : "prompts/sys_prompt.txt"
        },
        "base_prompt":{
            "content" : "prompts/tm_prompt.txt",
            "seed_topics" : ["International Relations", "War", "Peace", "Cooperation", "Countries"],
            "fs" : "prompts/few_shots.txt"
        },
        "merge_prompt":{
            "content" : "prompt/topic_merge.txt",
            "n_topics" : 50
        },
        "options" : {
            "temperature" : 0.0
            }
    }

    tm_agent = LLM_TopicModel(model=MODEL_NAME, settings=settings)

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:10]
    texts_list = df["text"].tolist()


    topics = tm_agent.get_topics(texts_list)
    print(f"Retrived topics: {topics}")

if __name__ == "__main__":
    main()
