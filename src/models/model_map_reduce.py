from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from database import Database
from utils.langchain import ExtendedHuggingFacePipeline
from .base_model import BaseModel


QUESTION_PROMPT_TEMPLATE = (
    "<Вопрос>: {question} <Контекст>: {context}"
)

COMBINE_PROMPT_TEMPLATE = (
    "<Вопрос>: {question} <Контекст>: Есть ответы на данный вопрос, из них надо выбрать наилучший. {summaries}"
)


class ModelMapReduce(BaseModel):
    def __init__(self, llm: ExtendedHuggingFacePipeline, database: Database, question_prompt=None, combine_prompt=None):
        super().__init__(llm, database, question_prompt=question_prompt, combine_prompt=combine_prompt)

    def init_prompt(self, question_prompt, combine_prompt, **kwargs):
        self.question_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=question_prompt or QUESTION_PROMPT_TEMPLATE
        )
        self.combine_prompt = PromptTemplate(
            input_variables=["summaries", "question"],
            template=combine_prompt or COMBINE_PROMPT_TEMPLATE
        )

    def init_qa_chain(self, **kwargs):
        self.qa_chain = load_qa_chain(
            self.llm, 
            chain_type="map_reduce", 
            question_prompt=self.question_prompt,
            combine_prompt=self.combine_prompt,
        )
