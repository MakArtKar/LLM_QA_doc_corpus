from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from database import Database
from models.base_model import BaseModel
from utils.langchain import ExtendedHuggingFacePipeline


PROMPT_TEMPLATE = (
    "<Вопрос>: {question} <Контекст>: {context}"
)


class ModelStuff(BaseModel):
    def __init__(self, llm: ExtendedHuggingFacePipeline, database: Database, prompt=None):
        super().__init__(llm, database, prompt=prompt)

    def init_prompt(self, prompt, **kwargs):
        prompt = prompt or PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt,
        )

    def init_qa_chain(self, **kwargs):
        self.qa_chain = load_qa_chain(
            self.llm, 
            chain_type="stuff", 
            prompt=self.prompt,
        )
