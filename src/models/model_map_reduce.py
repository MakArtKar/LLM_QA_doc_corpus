from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from database import Database
from .base_model import BaseModel


QUESTION_PROMPT_TEMPLATE = (
    "<SC6>Используя контекст ниже - найди всю информацию, релевантн ую вопросу в конце. "
    "Выведи всю информацию из контекста, релевантную вопросу в конце. "
    "Если в контексте нету никакой релевантной информации, то ничего не пиши. "
    "----------\n"
    "Контекст: {context}\n"
    "----------\n"
    "Вопрос: {question}\n"
    "Ответ: <extra_id_0>"
)

COMBINE_PROMPT_TEMPLATE = (
    "<SC6>Используя контекст ниже - ответь на вопрос."
    "Если не знаешь ответа - не пытайся угадать и просто напиши, что не знаешь ответа.\n"
    "Информация: {summaries}.\n"
    "Вопрос: {question}\n"
    "Ответ: <extra_id_0>"
)
                            


class ModelMapReduce(BaseModel):
    def __init__(self, database: Database, question_prompt=None, combine_prompt=None):
        super().__init__(database, question_prompt=question_prompt, combine_prompt=combine_prompt)

    def init_prompt(self, question_prompt, combine_prompt, **kwargs):
        self.question_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=question_prompt or QUESTION_PROMPT_TEMPLATE
        )
        self.combine_prompt = PromptTemplate(
            input_variables=["summaries", "question"],
            template=combine_prompt or COMBINE_PROMPT_TEMPLATE
        )

    def init_llm(self, **kwargs):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="Den4ikAI/FRED-T5-LARGE_text_qa",
            task="text2text-generation",
            model_kwargs={
                "temperature": 0.5,
                "max_length": 512,
            },
        )

    def init_qa_chain(self, **kwargs):
        self.qa_chain = load_qa_chain(
            self.llm, 
            chain_type="map_reduce", 
            question_prompt=self.question_prompt,
            combine_prompt=self.combine_prompt,
        )
