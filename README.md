# Topic modelling comparison suite
A project aimed at comparing the results of topic modelling tasks performed with various techniques:
* Gensim implementation of LDA (Latent Dirichlet Aloocation). 
* prodLDA: implemented.
* LLM_TopicModel (implemented): a generative AI powered topic model, directly outputting labels from input texts .
* BERTopic: https://github.com/MaartenGr/BERTopic .

The dataset used for this experiment is a corpus of the UN debates held throughout 1946-2023. 
Link: https://www.kaggle.com/datasets/namigabbasov/united-nations-general-debate-corpus-1946-2023 .

# Requirements
* Ollama installation to run LLM_topic modelling + pulling deepseek-r1:8b from ollama.
* packages in requirements.txt.

Having a GPU is reccomended to run LLM_TopicModel
