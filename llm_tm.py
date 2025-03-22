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
    def __init__(self, 
                 model, 
                 sys_p : str = None, 
                 base_p : str = None,
                 merge_p : str = None,
                 labelling_p : str = None,
                 n_topics : int = 2,
                 n_shots : str = "",
                 seed_topics : List[str] = "",
                 options : dict = None,
                 ):

        self.status = self._check_ollama_reqs(model)
        self.client = ollama.Client()
        self.model = model
        self.topics = None

        if sys_p:
            self.sys_prompt = self._get_prompt(sys_p)

        if labelling_p:
            self.labelling_prompt = self._get_prompt(labelling_p)

        if base_p:
            self.base_prompt = self._get_prompt(base_p)

        if merge_p:
            self.merge_prompt = self._get_prompt(merge_p)

        self.n_topics = n_topics
        self.opt = options if options else {}

        
        self.fs = self._get_prompt(n_shots) if n_shots else ""

        self.st = seed_topics if seed_topics else ""


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

        ########
        def extract_label(content, pattern = None):
            try:
                if self.model == "deepseek-r1:8b":
                    content = content.split("</think>")[1].strip()

                pattern = r"\*\*Label:\*\*\s*(.*)"
                result_1 = re.search(pattern, string=content)
                if result_1:
                    return result_1.group(1).strip()
                
                backup_pattern = r"\*\*\s*(.*)\s*\*\*"
                result_2 = re.search(backup_pattern, string=content)
                return result_2.group(1).strip() if result_2 else None

            except:
                print("Failed to extract topics. Continuing execution")
                return None
        ########

        self.client.chat(model=self.model, messages=[{"role" : "system", "content" : self.labelling_prompt}])

        labels = []
        for topic in topics:
            print(f"Labeling topic: {topic}")
            resp = self.client.generate(model=self.model, prompt=f"Assign a label to this topic represention in the requested format. Input: {topic}", options=self.opt)["response"]
            print(resp)
            label = extract_label(resp)
            print(f"Extracted: {label}")
            labels.append(label)
        return labels


    def get_topics(self, corpus : List[str], merge_limit = 60):
        
        ####
        def extract_topic(content, pattern = None):
            try:
                if self.model == "deepseek-r1:8b":
                    content = content.split("</think>")[1].strip() # making sure to only target the response

                if not pattern:
                    pattern = r"\*\*(.*?)\*\*"
                topics = re.findall(pattern, string=content)
            except IndexError:
                print("Failed to extract topics. Continuing execution")
                return None
            return topics
        ####

        raw_topics = []
        for i in range(len(corpus)):

            if len(raw_topics) >= merge_limit:
                print(f"Reached cutoff of {merge_limit} topics. Merging...")
                raw_topics = self.merge_topics(raw_topics)

            doc = corpus[i]
            interpolated_prompt = self.base_prompt.format(SEED_TOPICS=self.st, FEW_SHOTS=self.fs, BATCH=doc)
            print(f"\n\n{'-'*10}START OF RESPONSE N.{i+1}{'-'*10}")
            response = self.client.generate(model=self.model, prompt=interpolated_prompt, options=self.opt)["response"]
            print(response)
            print(f"\n{'-'*10}END OF RESPONSE{'-'*10}\n")
            topics = extract_topic(response)
            if i == 0 and not topics:
                continue
            print(topics)
            if topics:
                raw_topics.extend(topics)

        if not raw_topics:
            raise ValueError("No topics found. Try restaring the process or usng a different prompt.") 

        merged_topics = self.merge_topics(raw_topics)

        try:
            with open("data/results/llm_log.txt", "w", encoding="utf-8") as f:
                f.write(str(merged_topics))
        except:
            print("Failed to write topics to log")

        return raw_topics, merged_topics
    

    def merge_topics(self, topics : List[str]):

        ####
        def extract_labels(content, pattern = None):
            try:
                if self.model == "deepseek-r1:8b":
                    content = content.split("</think>")[1].strip()

                if not pattern:
                    pattern = r"\*\*(.*?)\*\*"
                labels = re.findall(pattern, string=content)

                return labels

            except Exception as e:
                print(f"Failed to extract labels due to {e}. Continuing execution")
                return None
         
        ####

        prompt = self.merge_prompt.format(NUM_TOPICS=self.n_topics, TOPICS_LIST=topics)
        topics = list(set(topics)) # dedupe
        print(f"Merging {len(topics)} topics")
        resp = self.client.generate(model=self.model, prompt=prompt, options=self.opt)["response"]
        print(resp)
        merged_topics = extract_labels(resp)
        print(f"Extracted: {merged_topics}")
        self.topics = merged_topics
        return merged_topics


    


def main():
    # mistral:7b-instruct-q4_K_M
    # deepseek-r1:8b
    MODEL_NAME = "deepseek-r1:8b"

    options =  {"temperature" : 0.0}

    tm_agent = LLM_TopicModel(model=MODEL_NAME,
                              base_p="prompts/tm_prompt.txt",
                              merge_p="prompts/topic_merge.txt",
                              labelling_p="prompts/topic_labelling.txt",
                              n_topics=10,
                              n_shots="prompts/few_shots.txt",
                              seed_topics=["International Relations", "War", "Peace", "Cooperation", "Countries"],
                              options=options)

    df = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")[:5]
    texts_list = df["text"].tolist()


    _, topics = tm_agent.get_topics(texts_list)
    print(f"Retrived topics: {topics}")
    

if __name__ == "__main__":
    main()
