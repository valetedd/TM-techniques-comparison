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
    def __init__(self, settings, model, sys : bool = False, labelling : bool = False):

        self.status = self._check_ollama_reqs(model)
        self.client = ollama.Client()
        self.model = model
        self.final_topics = None

        if sys:
            sys_p = settings["system_prompt"]["content"]
            self.sys_prompt = self._get_prompt(sys_p)

        if labelling:
            self.labelling_prompt = self._get_prompt(settings["labelling_prompt"]["content"])

        base_p = settings["base_prompt"]["content"]
        self.base_prompt = self._get_prompt(base_p)

        self.merge_prompt = self._get_prompt(settings["merge_prompt"]["content"])
        self.n_topics = settings["merge_prompt"]["n_topics"]

        try:
            few_shots = settings["base_prompt"]["fs"]
            self.fs = self._get_prompt(few_shots)
            

            self.st = settings["base_prompt"]["seed_topics"]
            self.opt = settings["options"]

        except KeyError:
            print("Few shots or seed topics not found in settings. Please provide them.")
            self.fs = ""
            self.st = ""  


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
    
    def label_topics(self, topics):

        self.client.chat(model=self.model, messages={"role" : "system", "content" : self.labelling_prompt})
        labels = []
        for topic in topics:
            print(f"Labeling topic: {topic}")
            resp = self.client.generate(model=self.model, prompt=f"{topic}. Label this topic", options=self.opt)["response"]
            print(resp)
            label = self._extract_label(resp)
            print(f"Extracted: {label}")
            labels.append(label)
        return labels


    def _extract_topic(self, content, pattern = None):
        try:
            content = content.split("</think>")[1] # making sure to only target the response

            if not pattern:
                pattern = r"(?:\d+\.|\-)\s\*\*(.*?)\*\*"
            topics = re.findall(pattern, string=content)
        except IndexError:
            print("No topics found")
            return None
        return topics


    def get_topics(self, corpus : List[str]):

        raw_topics = []
        for i in range(len(corpus)):
            doc = corpus[i]
            # print(batch)
            interpolated_prompt = self.base_prompt.format(SEED_TOPICS=self.st, FEW_SHOTS=self.fs, BATCH=doc)
            print(f"\n\n{'-'*10}START OF RESPONSE N.{i+1}{'-'*10}")
            response = self.client.generate(model=self.model, prompt=interpolated_prompt, options=self.opt)["response"]
            print(response)
            print(f"\n{'-'*10}END OF RESPONSE{'-'*10}\n")
            topics = self._extract_topic(response)
            if i == 0 and not topics:
                continue
            print(topics)
            raw_topics.extend(topics)

        merged_topics = self.final_topics = self.merge_topics(raw_topics)
        return raw_topics, merged_topics
    

    def merge_topics( self, topics : List[str]):
        
        prompt = self.merge_prompt.format(NUM_TOPICS=self.n_topics, TOPICS_LIST=topics)
        topics = list(set(topics)) # dedupe
        print(f"Merging {len(topics)} topics")
        resp = self.client.generate(model=self.model, prompt=prompt, options=self.opt)["response"]
        print(resp)
        merged_topics = self._extract_label(resp)
        print(f"Extracted: {merged_topics}")
        return merged_topics


    def _extract_label(self, content, pattern = None):
        try:
            content = content.split("</think>")[1].strip()

            if not pattern:
                pattern = r"^Topic\s+\d+\s*:$"
            clean_labels = re.sub(pattern, repl="", string=content)
            if "," in clean_labels:
                sep = ","
            elif "\n" in clean_labels:
                sep = "\n"
            labels = [match.split(":")[1].strip() for match in clean_labels.split(sep)]
        except IndexError:
            print("No topics found")
            return None
        return labels


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
            "content" : "prompts/topic_merge.txt",
            "n_topics" : 2
        },
        "options" : {
            "temperature" : 0.0
            }
    }

    tm_agent = LLM_TopicModel(model=MODEL_NAME, settings=settings)

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:2]
    texts_list = df["text"].tolist()


    _, topics = tm_agent.get_topics(texts_list)
    print(f"Retrived topics: {topics}")
    with open("data/results/llm_tm.txt", mode="w", encoding="utf-8") as f:
        f.write(str(topics))

if __name__ == "__main__":
    main()
