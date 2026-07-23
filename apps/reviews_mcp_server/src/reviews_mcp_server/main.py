from fastmcp import FastMCP
from qdrant_client import QdrantClient

from reviews_mcp_server.utils import process_reviews_context, retrieve_prefiltered_reviews_data

mcp = FastMCP("reviews_mcp_server")
qdrant_client = QdrantClient(url="http://qdrant:6333") # connecting from within docker network

@mcp.tool
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

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)