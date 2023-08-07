from abc import abstractmethod
from database import Database
from utils.langchain import ExtendedHuggingFacePipeline


class BaseModel:
    def __init__(self, llm: ExtendedHuggingFacePipeline, database: Database, **kwargs):
        self.llm = llm
        self.database = database
        self.retriever = self.database.faiss_db.as_retriever()

        self.init_prompt(**kwargs)
        self.init_qa_chain(**kwargs)

    @abstractmethod
    def init_prompt(self, **kwargs):
        raise NotImplementedError()
    
    @abstractmethod
    def init_qa_chain(self, **kwargs):
        raise NotImplementedError()

    def response(self, question: str, full: bool = False):
        docs = self.database.faiss_search(question)
        response = self.qa_chain({"input_documents": docs, "question": question})
        output_text = response['output_text']
        if not full:
            return output_text
        else:
            return output_text, docs
