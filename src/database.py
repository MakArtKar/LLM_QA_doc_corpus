import os
import pandas as pd
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from datetime import date
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv('DATA_PATH')

class Database:
    def __init__(self):
        self.data = pd.read_csv(data_path + "important.csv")
        self.faiss_db = None
        self.embeddings = SentenceTransformerEmbeddings(model_name="paraphrase-multilingual-mpnet-base-v2")
        self.initialize_faiss_db()

    def initialize_faiss_db(self):
        today = date.today().strftime("%Y-%m-%d")
        faiss_index_path = data_path + f"faiss_index_{today}"
        
        if os.path.exists(faiss_index_path):
            self.faiss_db = FAISS.load_local(faiss_index_path, self.embeddings)
        else:
            loader = CSVLoader(file_path=data_path + "important.csv")
            documents = loader.load()
            self.faiss_db = FAISS.from_documents(documents, self.embeddings)
            self.faiss_db.save_local(faiss_index_path)

    def faiss_search(self, query):
        if self.faiss_db is None:
            today = date.today().strftime("%Y-%m-%d")
            faiss_index_path = data_path + f"faiss_index_{today}"
            
            try:
                self.faiss_db = FAISS.load_local(faiss_index_path, self.embeddings)
            except:
                self.initialize_faiss_db()
        return self.faiss_db.similarity_search(query)
