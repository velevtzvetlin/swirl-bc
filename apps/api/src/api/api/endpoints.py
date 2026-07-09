from fastapi import APIRouter, Request
from api.api.models import RAGUsedContext, RagRequest, RagResponse
from api.agents.retrieval_generation import rag_pipeline_wrapper
from qdrant_client import QdrantClient

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rag_router = APIRouter()

qdrant_client = QdrantClient( # when we are running inside of docker network 
    url="http://qdrant:6333"
)

@rag_router.post("/")
def chat(
    request: Request,
    payload: RagRequest
) -> RagResponse:
    result = rag_pipeline_wrapper(payload.query)
    # transforming dictoinary into pydnatic schema
    return RagResponse(answer=result["answer"], used_context=[RAGUsedContext(**item) for item in result["used_context"]])

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["rag"])

