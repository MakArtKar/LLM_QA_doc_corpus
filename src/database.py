import os
import pandas as pd
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.schema import Document
from datetime import date
from dotenv import load_dotenv

load_dotenv()

data_path = os.getenv('DATA_PATH')

def convert_documents(documents_list):
    result_list = []
    for doc in documents_list:
        id_message = doc.page_content.split('id_message: ')[1].split('\n')[0]
        id_user = doc.page_content.split('id_user: ')[1].split('\n')[0]
        name = doc.page_content.split('name: ')[1].split('\n')[0]
        text = doc.page_content.split('text: ')[1]

        metadata = doc.metadata.copy()
        metadata.update({
            'id_message': id_message,
            'id_user': id_user,
            'name': name
        })

        new_doc = Document(page_content=text, metadata=metadata)

        result_list.append(new_doc)
    return result_list

class Database:
    def __init__(self, model_name=None):
        model_name = model_name or "paraphrase-multilingual-mpnet-base-v2"
        self.data = pd.read_csv(data_path + "important.csv")
        self.faiss_db = None
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
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

        ans = convert_documents(self.faiss_db.similarity_search(query, k=10))

        return ans