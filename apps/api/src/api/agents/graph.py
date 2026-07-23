from pydantic import BaseModel, Field
from typing import Annotated, List, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Prefetch, Document
from langgraph.checkpoint.postgres import PostgresSaver

from api.agents.tools import get_formatted_item_context, get_formatted_reviews_context
from api.agents.retrieval_generation import RAGUsedContext
from api.agents.agents import agent_node, intent_router_node
import json


from api.agents.agents import RAGUsedContext

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: List[RAGUsedContext] = [],
    trace_id: str = ""
    
### Edges
def tool_router(state: State) -> str:
    if state.final_answer:
        return "end"
    elif state.iteration > 2:
        return "end"
    if len(state.messages[-1].tool_calls) > 0: # if the model called a tool we want to go to the agent node again to see if it wants to call another tool
        return "tools"
    else:
        return "end"
    
def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"
    
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


### Workflow

workflow = StateGraph(State)
tools = [get_formatted_item_context, get_formatted_reviews_context]
tool_node = ToolNode(tools)


workflow.add_node("tool_node", tool_node)
workflow.add_node("agent_node", agent_node)
workflow.add_node("intent_router_node", intent_router_node)

workflow.add_edge(START, "intent_router_node")

workflow.add_conditional_edges(
    "intent_router_node",
    intent_router_conditional_edges,
    {
        "agent_node": "agent_node",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_node", # agent node conditionally routes to either tool node or end through tool router
    tool_router, 
    {
        "tools": "tool_node",
        "end": END
    }
)

workflow.add_edge("tool_node", "agent_node") # after the tool node we want to go back to the agent node to see if the model wants to call another tool

graph = workflow.compile()

### Agent Execution

def agent_stream_wrapper(question: str, thread_id: str) -> dict:
    def _string_for_sse(string):
        return f"data: {string}\n\n"
    
    def _process_graph_event(chunk):
        def _is_node_start(chunk):
            return chunk[1].get("type") == "task"

        def _tool_to_text(tool_call):
            if tool_call.get("name") == "get_formatted_item_context":
                return f"Looking for items: {tool_call.get('args').get('query', '')}."
            elif tool_call.get("name") == "get_formatted_reviews_context":
                return f"Fetching user reviews..."

        if _is_node_start(chunk):
            if chunk[1].get("payload", {}).get("name") == "intent_router_node":
                return "Analysing the question..."
            if chunk[1].get("payload", {}).get("name") == "agent_node":
                return "Planning..."
            if chunk[1].get("payload", {}).get("name") == "tool_node":
                input_state = chunk[1].get('payload', {}).get('input', None)
                if hasattr(input_state, 'messages'):
                    messages = input_state.messages
                elif isinstance(input_state, dict):
                    messages = input_state.get('messages', [])
                else:
                    messages = []
                last_msg = messages[-1] if messages else None
                tool_calls = getattr(last_msg, 'tool_calls', None) or [] if last_msg else []
                message = ". ".join([_tool_to_text(tool_call) for tool_call in tool_calls])
                return message
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration": 0
    }
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    with PostgresSaver.from_conn_string("postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db") as checkpointer:
        graph = workflow.compile(
            checkpointer=checkpointer

        )
        for chunk in graph.stream(
            initial_state, 
            config=config, 
            stream_mode=["debug", "values"]
        ):
            processed_chunk = _process_graph_event(chunk)
            # if the current chunk is not the response stream the processed_chunk will contain intermediate messages from the agent or tool nodes
            # the function will return string or None and bypass this stream if it is the response stream
            if processed_chunk:
                yield _string_for_sse(processed_chunk)
            
            if chunk[0] == "values":
                result = chunk[1]

    
    qdrant_client = QdrantClient(url="http://qdrant:6333")
    
    used_context = []
    
    for item in result.get("references", []):
        parent_asin = item.get("id")
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
                    "description": item.get("description", ""),
                    "parent_asin": parent_asin
                })

    yield _string_for_sse(
            json.dumps(
                {
                    "type": "final_answer",
                    "data": {
                                "answer": result.get("answer", ""),
                                "used_context": used_context,
                                "trace_id": result.get("trace_id", "")
                    }
                }
            )
    )
    
    
 