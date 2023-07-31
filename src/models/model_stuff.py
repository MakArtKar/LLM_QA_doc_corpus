from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from database import Database
from models.base_model import BaseModel


PROMPT_TEMPLATE = (
    "<SC6>Исопользуй контекст ниже, чтобы ответить на вопрос в конце. "
    "Если не знаешь ответа, не пытайся угадать его - просто скажи, что не знаешь ответ.\n"
    "----------\n"
    "Контекст: {context}\n"
    "----------\n"
    "Вопрос: {question}\n"
    "Ответ: <extra_id_0>"
)


class ModelStuff(BaseModel):
    def __init__(self, database: Database, prompt=None):
        super().__init__(database, prompt=prompt)

    def init_prompt(self, prompt, **kwargs):
        prompt = prompt or PROMPT_TEMPLATE
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt,
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
            chain_type="stuff", 
            prompt=self.prompt,
        )
