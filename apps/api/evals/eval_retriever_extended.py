
from api.agents.retrieval_generation import rag_pipeline

from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from langsmith import Client

from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings

from ragas.metrics.collections import Faithfulness, AnswerRelevancy

openai_client = AsyncOpenAI()
    
ls_client = Client()
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)
    
ragas_llm = llm_factory("gpt-4.1", client=openai_client, max_tokens=4000)
ragas_embeddings = OpenAIEmbeddings(client=openai_client, model="text-embedding-3-small")

# Precision in AI measures the accuracy of a model's positive predictions. 
# It calculates the exact percentage of positive results that were genuinely correct out of all the positive predictions the model made. 
# A higher precision means fewer false alarms.
# The Google Machine Learning Crash Course defines the mathematical formula for precision as:
# \(\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}\)Where:True Positives (TP): 
# The model correctly predicted a positive outcome.False Positives (FP): The model incorrectly predicted a positive outcome (a false alarm).
# Why Precision MattersPrecision is vital when the cost of a false positive is high. For instance, in Customer Churn Models, 
# if an AI incorrectly predicts a customer will leave (a false positive), a company might waste resources giving them unnecessary discounts. Similarly, 


#retrieved context ids and reference context ids / retrieved context ids -> precision
# Here's the code:
#
# score = (
#     len(retrieved_context_ids & reference_context_ids)
#     / len(retrieved_context_ids)
#     if retrieved_context_ids else 0.0
# )
#
# Let's break it down.
#
# Suppose:
#
# retrieved_context_ids = {1, 2, 3, 4, 5}
# reference_context_ids = {2, 4, 6}
#
# The intersection is:
#
# retrieved_context_ids & reference_context_ids
# # {2, 4}
#
# So:
#
# precision = 2 / 5 = 0.4
def context_prescision_id_based(run, example): #run.output is the actual trace in langsmith
    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]} # builds a set of retrieved context IDs
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]} # builds a set of reference context IDs
    score = len(retrieved_context_ids & reference_context_ids) / len(retrieved_context_ids) if retrieved_context_ids else 0.0
    return score

def context_recall_id_based(run, example):
    retrieved_context_ids = {str(id) for id in run.outputs["retrieved_context_ids"]} # builds a set of retrieved context IDs
    reference_context_ids = {str(id) for id in example.outputs["reference_context_ids"]} # builds a set of reference context IDs
    scorer = len(retrieved_context_ids & reference_context_ids) / len(reference_context_ids) if reference_context_ids else 0.0
    return scorer

def ragas_faithfulness(run):
    scorer = Faithfulness(llm=ragas_llm)
    
    result = scorer.score(
       user_input=run.outputs["question"], response=run.outputs["answer"], retrieved_contexts=run.outputs["retrieved_context"]
    )
    
    return result.value

def ragas_relevancy(run):
    scorer = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    result = scorer.score(
        user_input=run.outputs["question"], 
        response=run.outputs["answer"]
    )
    return result.value


# fetch the test cases
# execute the rag_pipeline with the question in the test 
# example will come directly from the datset via langsmith client 

print("Evaluating plain retriever")
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid=False, rerank=False), #input of reference dataset iteraetion (injecting question into rag pipeline)
    data='rag-evaluation-dataset-extended',
    evaluators=[
        context_prescision_id_based, 
        context_recall_id_based,
        # ragas_faithfulness,
        # ragas_relevancy,
    ],
    experiment_prefix='plain',
    max_concurrency=10
)

print("Evaluating hybrid retriever")
results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid=True, rerank=False), #input of reference dataset iteraetion (injecting question into rag pipeline)
    data='rag-evaluation-dataset-extended',
    evaluators=[
        context_prescision_id_based, 
        context_recall_id_based,
        # ragas_faithfulness,
        # ragas_relevancy,
    ],
    experiment_prefix='hybrid',
    max_concurrency=10
)

# print("Evaluating hybrid retriever with reranking")
# results = ls_client.evaluate(
#     lambda x: rag_pipeline(x["question"], qdrant_client, topk_k=10, hybrid=True, rerank=True), #input of reference dataset iteraetion (injecting question into rag pipeline)
#     data='rag-evaluation-dataset-extended',
#     evaluators=[
#         context_prescision_id_based, 
#         context_recall_id_based,
#         ragas_faithfulness,
#         ragas_relevancy,
#     ],
#     experiment_prefix='hybrid-rerank'
# )