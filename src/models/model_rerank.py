from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.output_parsers import RegexParser

from database import Database
from .base_model import BaseModel


output_parser = RegexParser(
    regex=r"(.*?)\nОценка: (.*)",
    output_keys=["answer", "score"],
    default_output_key="answer",
)

PROMPT_TEMPLATE = (
    "<SC6>Используй контекст ниже, чтобы ответить на вопрос в конце. "
    # "Если не знаешь ответа, не пытайся угадать его - просто скажи, что не знаешь ответ. "
    "В дополнение к ответу на вопрос также выведи оценку того, насколько хорошо дан ответ на вопрос от 0 до 100. "
    "Ответ должен быть в следующем формате:\n\n"
    "Вопрос: [здесь вопрос]\n"
    "Ответ: [здесь ответ]\n"
    "Оценка: [оценка]\n\n"
    "Контекст: {context}\n"
    "Вопрос: {question}\n"
    "Ответ и оценка ответа: <extra_id_0>"
)


class ModelRerank(BaseModel):
    def __init__(self, database: Database, prompt=None):
        super().__init__(database, prompt=prompt)

    def init_prompt(self, prompt, **kwargs):
        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt or PROMPT_TEMPLATE,
            output_parser=output_parser,
        )

    def init_llm(self, **kwargs):
        self.llm = HuggingFacePipeline.from_model_id(
            model_id="ai-forever/FRED-T5-large",
            task="text2text-generation",
            model_kwargs={
                "temperature": 0.5,
                "max_length": 512,
            },
        )

    def init_qa_chain(self, **kwargs):
        self.qa_chain = load_qa_chain(
            self.llm, 
            chain_type="map_rerank", 
            prompt=self.prompt,
        )
