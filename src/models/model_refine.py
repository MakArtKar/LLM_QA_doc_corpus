from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from database import Database
from utils.langchain import ExtendedHuggingFacePipeline
from .base_model import BaseModel

INITIAL_PROMPT_TEMPLATE = (
    "<Вопрос>: {question} <Контекст>: {context_str}"
)

REFINE_PROMPT_TEMPLATE = (
    "<Вопрос>: {question} <Контекст>: "
    "У нас есть ответ на него: {existing_answer}\n"
    "Есть возможность улучшить существующий ответ, (но только, если текущий ответ не подходит) с помощью контекста ниже.\n"
    "----------\n"
    "{context_str}\n"
    "----------\n"
    "Имея новый контекст, уточни существующий ответ, чтобы лучше ответить на изначальный вопрос. Ответь как можно более развернуто, сохранив как можно больше информации. "
    "Если контекст бесполезный - верни изначальный ответ.\n"
)


class ModelRefine(BaseModel):
    def __init__(self, llm: ExtendedHuggingFacePipeline, database: Database, initial_prompt=None, refine_prompt=None):
        super().__init__(llm, database, initial_prompt=initial_prompt, refine_prompt=refine_prompt)

    def init_prompt(self, initial_prompt, refine_prompt, **kwargs):
        self.initial_prompt = PromptTemplate(
            input_variables=["context_str", "question"],
            template=initial_prompt or INITIAL_PROMPT_TEMPLATE
        )
        self.refine_prompt = PromptTemplate(
            input_variables=["question", "existing_answer", "context_str"],
            template=refine_prompt or REFINE_PROMPT_TEMPLATE
        )

    def init_qa_chain(self, **kwargs):
        self.qa_chain = load_qa_chain(
            self.llm, 
            chain_type="refine", 
            question_prompt=self.initial_prompt,
            refine_prompt=self.refine_prompt,
        )
