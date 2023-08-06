from transformers import pipeline
from utils.langchain import ExtendedHuggingFacePipeline

def get_by_model_id(model_id, task, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    if task == 'question-answering':
        return get_by_pipeline(pipeline(
            task=task,
            model=model_id,
            tokenizer=model_id,
        ))
    return ExtendedHuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        model_kwargs=model_kwargs,
    )


def get_by_pipeline(pipeline):
    return ExtendedHuggingFacePipeline(pipeline=pipeline)
