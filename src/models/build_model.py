from .model_stuff import ModelStuff
from .model_refine import ModelRefine
from .model_map_reduce import ModelMapReduce
from .model_rerank import ModelRerank
from llms.base import get_by_model_id


MODELS = {
    'stuff': ModelStuff,
    'refine': ModelRefine,
    'map_reduce': ModelMapReduce,
    'rerank': ModelRerank,
}

def get_model(database, strategy, model_id, task, model_kwargs, **kwargs):
    llm = get_by_model_id(model_id, task, model_kwargs)
    if strategy == 'rerank':
        llm.with_score = True
    return MODELS[strategy](llm, database, **kwargs)