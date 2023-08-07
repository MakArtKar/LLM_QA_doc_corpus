import os

from transformers import pipeline, BitsAndBytesConfig
from utils.langchain import ExtendedHuggingFacePipeline


def prepare_model_kwargs(model_kwargs):
    model_kwargs.update({'use_auth_token': os.environ['HUGGINGFACEHUB_API_TOKEN']})
    if 'quantization_config' in model_kwargs:
        model_kwargs['quantization_config'] = BitsAndBytesConfig(**model_kwargs['quantization_config'])


def get_by_model_id(model_id, task, model_kwargs=None, pipeline_kwargs=None):
    print((
        "\nModel building started\n"
        f"Model: {model_id}"
    ))
    model_kwargs = model_kwargs or {}
    prepare_model_kwargs(model_kwargs)
    pipeline_kwargs = pipeline_kwargs or {}
    if task == 'question-answering':
        model =  get_by_pipeline(pipeline(
            task=task,
            model=model_id,
            tokenizer=model_id,
            model_kwargs=model_kwargs,
            **pipeline_kwargs,
        ))
    else:
        model = ExtendedHuggingFacePipeline.from_model_id(
            model_id=model_id,
            task=task,
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
        )
    print("\nModel building ended\n")
    return model


def get_by_pipeline(pipeline):
    return ExtendedHuggingFacePipeline(pipeline=pipeline)
