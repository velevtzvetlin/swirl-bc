import openai
from langsmith import traceable, get_current_run_tree

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
def retrieve_data(query, qdrant_client, k=5): # 5 most similar items to users query
    embedding = get_embedding(query) # so we are actually creating related vector here
    
    results = qdrant_client.query_points(
        collection_name="Amazon-items-collection-01",
        query=embedding,
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
    prompt = f"""You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- Answer the question based on the provided context only.
- Never use word context and refer to it as the available products.
- Do not use markdown formatting.

Context:
{preprocessed_context}

Question:
{question}
"""
    return prompt

@traceable(
    name="generate_answer",
    run_type="llm", # we are running an llm call
    metadata={"ls_provider": "openai", "model": "gpt-5-nano"} ## langsmith needs this to calculate cost of runs
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": prompt}
        ],
        reasoning_effort="minimal"
    )
    current_run  = get_current_run_tree()
    if current_run: # of tokens used for input and how many used for output
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return response.choices[0].message.content

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, topk_k=5):
    retrievedContext = retrieve_data(question, qdrant_client, k=topk_k)
    processed_context = process_context(retrievedContext)
    prompt = build_prompt(processed_context, question)
    answer = generate_answer(prompt) ## llm call w/ prompt
    
    final_answer = {
        "answer": answer,
        "question": question,
        "retrieved_context_ids": retrievedContext["retrieved_context_ids"],
        "retrieved_context": retrievedContext["retrieved_context"],
        "similarity_scores": retrievedContext["similarity_scores"]
    }
    
    return final_answer