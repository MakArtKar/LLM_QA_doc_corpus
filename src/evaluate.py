import pandas as pd


def evaluate(model, ground_truths):
    questions = ground_truths['question']
    model_answers = [model.respond(question) for question in questions]

    ground_answers = ground_truths['ground_truth_answer']
    model_evaluation = pd.DataFrame({'model_answer': model_answers, 'correct_answer': ground_answers})
    return model_evaluation


important = pd.read_csv("../data/important.csv")
print(important["name"])
g_t = pd.read_csv("../data/ground_truths.csv", sep=';')
print(g_t[["question", "ground_truth_answer"]])

model = ...
evaluate(model=model, ground_truths=g_t)
