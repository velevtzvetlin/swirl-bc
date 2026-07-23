from fastmcp import FastMCP
from qdrant_client import QdrantClient

from items_mcp_server.utils import retrieve_items_data, rerank_data, process_context

mcp = FastMCP("items_mcp_server")
qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network

@mcp.tool
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
 
    retrievedContext = retrieve_items_data(
        query, 
        qdrant_client, 
        k=20
    )
    # if we set rerank to True, we re-order the retrieved context based on the reranker results
    retrievedContext = rerank_data(query, retrievedContext, top_k=top_k)
    formatted_context = process_context(retrievedContext)
    return formatted_context

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)