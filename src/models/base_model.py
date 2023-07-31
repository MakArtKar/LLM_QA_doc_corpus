from abc import abstractmethod
from database import Database


class BaseModel:
    def __init__(self, database: Database, **kwargs):
        self.database = database
        self.retriever = self.database.faiss_db.as_retriever()

        self.init_prompt(**kwargs)
        self.init_llm(**kwargs)
        self.init_qa_chain(**kwargs)

    @abstractmethod
    def init_prompt(self, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def init_llm(self, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def init_qa_chain(self, **kwargs):
        raise NotImplementedError()

    def get_relevant_documents(self, question: str):
        return self.retriever.get_relevant_documents(query=question)

    def response(self, question: str):
        docs = self.get_relevant_documents(question)
        return self.qa_chain({"input_documents": docs, "question": question}, return_only_outputs=True)
