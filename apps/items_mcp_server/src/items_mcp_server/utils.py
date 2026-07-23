import cohere
import openai
from qdrant_client.models import Prefetch, Document
from qdrant_client import models

def get_embedding(text):
    response = openai.embeddings.create( # was client before
        input=text,
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding

def retrieve_items_data(query, qdrant_client, k=5, hybrid=True): # 5 most similar items to users query
    embedding = get_embedding(query) # so we are actually creating related vector here
    
    if hybrid:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            prefetch=[
                Prefetch(
                    query=embedding,
                    using='text-embedding-3-small', #name of the vector in the collection
                    limit=20
                ), 
                Prefetch(
                    query=Document(text=query, model='qdrant/bm25'),
                    using='bm25', #name of the vector in the collection
                    limit=20
                )
            ],
            # can set specific weights on how much importance we put on dense embeddings and bm25 lexical retrievals
            # in this case dense vector retrival is weighted as 3 times more important than bm25
            # if user queries are more keyword-based, you might want to increase the weight of bm25
            # if more contextual then you might want to increase the weight of the dense vector retrieval
            # weights can be added as params to the retrieve data function
            query=models.RrfQuery(rrf=models.Rrf(weights=[3,1])), # how you fuse the results....
            limit=k # once fused we are returning top k results
        )
    else:
        results = qdrant_client.query_points(
            collection_name="Amazon-items-collection-01-hybrid-search",
            query=embedding,
            using='text-embedding-3-small',
            limit=k
        )
    
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    retrieved_context_ratings = []
    
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload.get("preprocessed_description"))
        similarity_scores.append(result.score)
        retrieved_context_ratings.append(result.payload.get("average_rating"))
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores,
        "retrieved_context_ratings": retrieved_context_ratings
    }

def rerank_data(query, context, topk_k=5):
    cohere_client = cohere.ClientV2()
    # so we pass the query and the array of preprocessed_description name, desc context chunks
    response = cohere_client.rerank(
        model="rerank-v3.5",
        query=query,
        documents=context['retrieved_context'],
        top_n=topk_k
    )
    order = [result.index for result in response.results]
    # re order the original context
    return {
        "retrieved_context_ids": [context["retrieved_context_ids"][i] for i in order],
        "retrieved_context": [context["retrieved_context"][i] for i in order],
        "similarity_scores": [context["similarity_scores"][i] for i in order],
        "retrieved_context_ratings": [context["retrieved_context_ratings"][i] for i in order]
    }

def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context  