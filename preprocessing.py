import spacy
model_name = "en_core_web_md"
if not spacy.util.is_package(model_name):
    print(f"Model '{model_name}' is not installed. Please install it using:")
    print(f"python -m spacy download {model_name}")

from nltk.corpus import stopwords

import pandas as pd
from pathlib import Path
import os
import pickle

import matplotlib.pyplot as plt
from typing import List, Callable, Iterable, Optional, Literal

from gensim.corpora import Dictionary
from gensim.models import Phrases



def basic_pp(corpus: list[str], n_grams : Optional[Literal["bi-grams", "tri-grams"]] = None) -> list[list[str]]:

    nlp = spacy.load("en_core_web_md", exclude=["tok2vec", "parser"])
    sw = frozenset(stopwords.words("english"))

    processed_corpus = []
    for i, doc in enumerate(nlp.pipe(texts=corpus, batch_size=100)):
        if i % 100 == 0:
            print(f"Processing document {i} out of {len(corpus)}")
            
        tokens = [t.lemma_ for t in doc
                    if t.is_alpha and not 
                    t.pos_ in {"ADV"} and not
                    (t.text in sw or len(t.text) <= 2)]
        
        processed_corpus.append(tokens)

    if n_grams:
        bi_min_count = max(5, len(corpus) // 1000) 
        bigram = Phrases(processed_corpus, 
                         min_count=bi_min_count, 
                         threshold=0.5, 
                         scoring="npmi")

        if n_grams == "tri-grams":
            tri_min_count = max(3, len(corpus) // 1500)  # Lower threshold for trigrams
            trigram = Phrases(bigram[processed_corpus], 
                              min_count=tri_min_count, 
                              threshold=0.5, 
                              scoring="npmi")

            processed_corpus = [trigram[bigram[doc]] for doc in processed_corpus]

        else:
            processed_corpus = [bigram[doc] for doc in processed_corpus]
            

    return processed_corpus


def BOW_pp(
        docs: List[str] | List[List[str]],
        from_preprocessed : bool = True,
        filter_extr : bool = False,
        exclusion_floor : int = 0,
        exclusion_ceil : float = 0.4):

    if not from_preprocessed:
        docs = basic_pp(docs)

    dictionary = Dictionary(docs)
    
    if filter_extr:
        total_docs = len(docs)
        exclusion_floor = exclusion_floor if exclusion_floor else max(5, int(total_docs*0.001))
        dictionary.filter_extremes(no_below=exclusion_floor, no_above=exclusion_ceil)

    bow_repr = [dictionary.doc2bow(doc) for doc in docs]

    return dictionary, bow_repr


def get_pp_data(data_path:str, func:Callable, **kwargs):
    df = pd.read_csv(filepath_or_buffer=data_path)
    texts_list = df["text"].tolist()
    pp_data = func(texts_list, **kwargs)
    return pp_data


def load_pp(folder : str | Path, file_or_files : str | Iterable):
    if not isinstance(folder, (str, Path)):
        raise TypeError("'folder' must be a str or a Path object")
    path = Path(folder) if isinstance(folder, str) else folder

    files = []
    for f in path.iterdir():

        if f.name not in file_or_files:
            continue
        
        path = os.path.join(folder, f.name)

        if f.name.endswith("pkl"):
            with open(path, mode="rb") as f:
                file = pickle.load(f)
            files.append(file)

        elif f.name.endswith("dict"):
            file = Dictionary.load(path)
            files.append(file)
        
    if len(files) != len(file_or_files):
        raise ValueError("Not all specified files were found in the folder")

    return tuple(files)
        

def main():

    data = pd.read_csv(filepath_or_buffer="data/UN_speeches/UNGDC_1946-2023.csv")["text"].tolist()
    pp = basic_pp(data, n_grams="tri-grams")
    # pp = load_pp(folder="data/UN_PP", file_or_files="tokenized.pkl")["tokenized"]
    dct, bow = BOW_pp(docs=pp, filter_extr=True, from_preprocessed=True)
    
    dct.save("data/UN_PP/dictionary.dict")
    with open("data/UN_PP/tokenized.pkl", mode="wb") as f:
        pickle.dump(pp, f)
    with open("data/UN_PP/bow.pkl", mode="wb") as f1:
        pickle.dump(bow, f1)

if __name__ == "__main__":
    main()
