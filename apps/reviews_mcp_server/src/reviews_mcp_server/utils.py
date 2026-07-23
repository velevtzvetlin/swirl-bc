
import openai

from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchAny, Prefetch

def get_embedding(text):
    response = openai.embeddings.create( # was client before
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def process_reviews_context(context):
    formatted_context = ""
    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, user review: {chunk}\n"
    return formatted_context   

def retrieve_prefiltered_reviews_data(query, parent_asins, qdrant_client, top_k=5): # 5 most similar items to users query
    embedding = get_embedding(query) # so we are actually creating related vector here
    
    results = qdrant_client.query_points(
        collection_name="Amazon-reviews-collection-01",
        prefetch=[
            Prefetch(
                query=embedding,
                using='text-embedding-3-small', #name of the vector in the collection
                filter=Filter( 
                    must=[
                        FieldCondition(
                            key="parent_asin",
                            match=MatchAny(
                                any=parent_asins
                            )
                        )
                    ]
                ),
                limit=20
            )
        ],
        query=FusionQuery(fusion="rrf"),
        limit=top_k
    )
    
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []
    
    for result in results.points:
        retrieved_context_ids.append(result.payload["parent_asin"])
        retrieved_context.append(result.payload.get("preprocessed_data", ""))
        similarity_scores.append(result.score)
    
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }