from ast import MatchValue
from tracemalloc import Filter

import openai
import cohere
from langsmith import traceable, get_current_run_tree
import instructor
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document


from api.agents.utils.prompt_management import prompt_template_config

class RAGUsedContext(BaseModel):
    id: str = Field(description="ID of item used to answer the question")
    description: str = Field(description="Description of item used to answer the question")

class RAGGenerationResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of items used to answer the question")
    

@traceable(
    name="embed_query",
    run_type="embedding", # tells langsmith we are actually running embedding text -> specific set of vectors
    metadata={
        "ls_provider": "openai", 
              "model": "text-embedding-3-small"
    } ## langsmith needs this to calculate cost of runs
)
def get_embedding(text):
    response = openai.embeddings.create( # was client before
        input=text,
        model="text-embedding-3-small"
    )
    
    current_run  = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.data[0].embedding

@traceable(
    name="retrieve_data",
    run_type="retriever"
)
def retrieve_data(query, qdrant_client, k=5, hybrid=True): # 5 most similar items to users query
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

@traceable(
    name="rerank_data",
    run_type="tool"
)
def rerank_data(query, context, topk_k=5):
    cohere_client = cohere.ClientV2()
    # so we pass the query and the array of preprocessed_description name, desc context chunks
    response = cohere_client.rerank(
        model="rerank-english-v4-pro",
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
    
@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context

@traceable(
    name="build_prompt",
    run_type="prompt"
)
def build_prompt(preprocessed_context, question): # question is the users question...
    template = prompt_template_config("api/agents/prompts/retrieval_generation.yaml", "retrieval_generation")
    prompt = template.render(preprocessed_context=preprocessed_context, question=question)
    return prompt

@traceable(
    name="generate_answer",
    run_type="llm", # we are running an llm call
    metadata={"ls_provider": "openai", "model": "gpt-5-nano"} ## langsmith needs this to calculate cost of runs
)
def generate_answer(prompt):
    client = instructor.from_provider("openai/gpt-5.4-nano", mode=instructor.Mode.RESPONSES_TOOLS) # can only add reasoning effort with the responses api

    response, raw_response = client.create_with_completion(
        messages=[
            {"role": "system", "content": prompt}
        ],
        response_model=RAGGenerationResponse,
        reasoning={"effort": "none"}
    )
    current_run  = get_current_run_tree()
    if current_run: # of tokens used for input and how many used for output
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }
    return response

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, topk_k=5, hybrid=True, rerank=False, retrieve_k=20):
    retrievedContext = retrieve_data(
        question, 
        qdrant_client, 
        k=retrieve_k if rerank else topk_k,
        hybrid=hybrid
    )
    if rerank:
        # if we set rerank to True, we re-order the retrieved context based on the reranker results
        retrievedContext = rerank_data(question, retrievedContext, topk_k=topk_k)
    processed_context = process_context(retrievedContext)
    prompt = build_prompt(processed_context, question)
    answer = generate_answer(prompt) ## llm call w/ prompt
    final_answer = {
        "answer": answer.answer,
        "references": answer.references,
        "question": question,
        "retrieved_context_ids": retrievedContext["retrieved_context_ids"],
        "retrieved_context": retrievedContext["retrieved_context"],
        "similarity_scores": retrievedContext["similarity_scores"],
    }
    
    return final_answer

def get_point_by_parent_asin(parent_asin, qdrant_client):
    return qdrant_client.scroll(
        collection_name="Amazon-items-collection-01-hybrid-search",
        with_payload=True,
        with_vectors=False,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="parent_asin",
                    match=MatchValue(value=parent_asin)
                )
            ]
        )
    )[0][0].payload

def rag_pipeline_wrapper(question, topk_k=5):
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    result =  rag_pipeline(question, qdrant_client, topk_k=topk_k)
    
    used_context = []
    
    for item in result.get("references", []):
        parent_asin = item.id
        if parent_asin:
            payload = get_point_by_parent_asin(parent_asin, qdrant_client)
            print(payload)
            image_url = payload.get("image", "")
            price = payload.get("price")
            # so image_url, price comes from the qdrant query after we receive the used references
            # item description is actually coming back from the references in the LLM answer where we passed the expected object shape to get that
            if image_url:
                used_context.append({
                    "image_url": image_url,
                    "price": price,
                    "description": item.description,
                    "parent_asin": parent_asin
                })
                
    return {
        "answer": result["answer"],
        "used_context": used_context
    }
    
    
    
    
