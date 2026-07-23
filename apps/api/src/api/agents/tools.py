import openai
import cohere
from qdrant_client import QdrantClient
from langsmith import traceable, get_current_run_tree
from qdrant_client.models import FieldCondition, Filter, FusionQuery, MatchAny, Prefetch, Document
from qdrant_client import models
from langchain_core.tools import tool

@traceable(
    name="embed_query",
    run_type="embedding", # tells langsmith we are actually running embedding text -> specific set of vectors
    metadata={
        "ls_provider": "openai", 
        "ls_model_name": "text-embedding-3-small"
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
    name="retrieve_items_data",
    run_type="retriever"
)
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

@traceable(
    name="rerank_data",
    run_type="tool"
)
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
    
@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for id, chunk, rating in zip(context["retrieved_context_ids"], context["retrieved_context"], context["retrieved_context_ratings"]):
        formatted_context += f"- ID: {id}, rating: {rating}, description: {chunk}\n"
    return formatted_context   

@tool
def get_formatted_item_context(query: str, top_k: int = 5) -> str:
    """
        Search available products and return the top k matching inventory items.

        Expand the customer's question into 1–5 concise search statements and issue them in parallel in a single turn. Each statement covers one distinct product or attribute; no two may express the same intent. Use natural product-description language. If no brand or model is specified, search broadly rather than refusing.

        "Earphones for me and a waterproof speaker"
        -> "Personal earphones" | "Waterproof speaker"
        "A warm winter jacket for hiking"
        -> "Insulated winter jacket" | "Hiking outerwear for cold weather"

        Before calling, check what earlier calls in this conversation already returned. Search only for what is missing; results already retrieved remain valid and must not be fetched again.

        Args:
        query: A single search statement describing one product or attribute.
        top_k: Number of items to retrieve. Works best with 5 or more.

        Returns:
        A string of the top k available products, each prefixed with its ID and average rating.
    """
    qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network
    retrievedContext = retrieve_items_data(
        query, 
        qdrant_client, 
        k=20
    )
    # if we set rerank to True, we re-order the retrieved context based on the reranker results
    retrievedContext = rerank_data(query, retrievedContext, topk_k=top_k)
    formatted_context = process_context(retrievedContext)
    return formatted_context


# reviews retrieval metadata tool
@traceable(
    name="retrieve_prefiltered_reviews_data",
    run_type="retriever"
)
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

@traceable(
    name="format_retrieved_context",
    run_type="prompt"
)
def process_reviews_context(context):
    formatted_context = ""
    for id, chunk in zip(context["retrieved_context_ids"], context["retrieved_context"]):
        formatted_context += f"- ID: {id}, user review: {chunk}\n"
    return formatted_context   

@tool
def get_formatted_reviews_context(query: str, parent_asins: list[str], top_k: int = 5) -> str:
    """
        Get the top k reviews matching a query for a list of prefiltered items.

        Args:
            query: The query to get the top k reviews for
            item_list: The list of item IDs to prefilter for before running the query
            top_k: The number of reviews to retrieve, this should be at least 20 if multiple items are prefiltered

        Returns:
            A string of the top k context chunks with IDs prepending each chunk, each representing a review for a given inventory item for
    """

    qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network
    retrievedContext = retrieve_prefiltered_reviews_data(
        query,
        parent_asins,
        qdrant_client,
        top_k=top_k
    )
    # if we set rerank to True, we re-order the retrieved context based on the reranker results
    formatted_context = process_reviews_context(retrievedContext)
    return formatted_context