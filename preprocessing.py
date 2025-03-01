import spacy
model_name = "en_core_web_sm"
if not spacy.util.is_package(model_name):
    print(f"Model '{model_name}' is not installed. Please install it using:")
    print(f"python -m spacy download {model_name}")

from nltk.corpus import stopwords

import pandas as pd
import string

import matplotlib.pyplot as plt
from typing import List, Callable

from gensim.corpora import Dictionary



def basic_pp(corpus: list[str]) -> list[list[str]]:

    tokenizer = spacy.load("en_core_web_sm")
    sw = set(stopwords.words("english"))

    data = []
    for doc in corpus:
        doc = doc.translate(str.maketrans("", "", string.punctuation + string.digits + "\n")) # removing punctuation (X:insertions; Y:to-be-replaced; Z:deletions)
        data.append(
            [t.text for t in tokenizer(doc)
                    if not (t.text in sw
                    or len(t.text) <= 2)]
                            )

    return data


def LDA_pp(docs: List[List[str]]):

    dictionary = Dictionary(docs)

    bow_repr = [dictionary.doc2bow(doc) for doc in docs]
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    return dictionary, bow_repr


def write_pp(data: list[list[str]]):
    pass



def get_pp_data(data_path:str, func:Callable):
    df = pd.read_csv(filepath_or_buffer=data_path)
    texts_list = df["text"].tolist()
    pp_data = func(texts_list)
    return pp_data


