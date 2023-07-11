# import pandas as pd
# from sentence_transformers import SentenceTransformer, util
# import torch
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader

class Database:
    def __init__(self, csv_file_path):
        # self.data = pd.read_csv(csv_file_path)
        # self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        # self.embeddings = self.model.encode(self.data['text'])
        # self.messages = list(self.data['text'])
        self.faiss_db = None

    # def add_message(self, message):
    #     self.messages.append(message)
    #     new_embedding = self.model.encode([message])
    #     self.embeddings = torch.cat([self.embeddings, new_embedding])
    #     if self.faiss_db is not None:
    #         self.faiss_db.add_document(message)

    def initialize_faiss_db(self):
        loader = CSVLoader(file_path="/home/makart/data/important.csv")
        documents = loader.load()
        embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
        self.faiss_db = FAISS.from_documents(documents, embeddings)

    def faiss_search(self, query):
        if self.faiss_db is None:
            self.initialize_faiss_db()
        return self.faiss_db.similarity_search(query)

    def merge_db(self, other_db):
        self.add_messages(other_db.messages)
