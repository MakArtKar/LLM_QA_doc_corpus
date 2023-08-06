from typing import Any, List, Optional
from langchain.llms.huggingface_pipeline import HuggingFacePipeline, VALID_TASKS
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.utils import enforce_stop_tokens
from langchain.output_parsers import RegexParser


VALID_TASKS = (*VALID_TASKS, "question-answering")


class ExtendedHuggingFacePipeline(HuggingFacePipeline):
    with_score: bool = False

    qa_input_parser = RegexParser(
        regex=r"<Вопрос>: (.*?) <Контекст>: (.*)",
        output_keys=["question", "context"],
        default_output_key="question",
    )
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
 
        if self.pipeline.task == "question-answering":
            prompt = self.qa_input_parser.parse(prompt.replace('\n', ' '))
        print(prompt)
        print('=' * 20)
        response = self.pipeline(prompt)

        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            text = response[0]["generated_text"][len(prompt) :]
        elif self.pipeline.task == "text2text-generation":
            text = response[0]["generated_text"]
        elif self.pipeline.task == "summarization":
            text = response[0]["summary_text"]
        elif self.pipeline.task == "question-answering":
            if self.with_score:
                score = int(float(response['score']) * 100)
                text = f"<Answer>: {response['answer']}\n<Score>: {score}"
            else:
                text = response['answer']
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        
        if stop:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, stop)
        return text
