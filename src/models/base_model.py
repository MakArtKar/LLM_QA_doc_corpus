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

    def get_relevant_documents(self, question: str):
        return self.retriever.get_relevant_documents(query=question)

    def response(self, question: str, full: bool = False):
        docs = self.get_relevant_documents(question)
        response = self.qa_chain({"input_documents": docs, "question": question})
        print(response)
        print('^' * 20)
        if not full:
            response = response['output_text']
            return response
        raise NotImplementedError('BaseModel.response with full = True is not implemented')
