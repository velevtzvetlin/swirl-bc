from fastapi import APIRouter, Request
from api.api.models import RAGUsedContext, RagRequest, RagResponse
from api.agents.graph import agent_wrapper

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def chat(
    request: Request,
    payload: RagRequest
) -> RagResponse:
    result = agent_wrapper(payload.query)
    # transforming dictoinary into pydnatic schema
    return RagResponse(answer=result.get("answer", ""), used_context=[RAGUsedContext(**item) for item in result.get("used_context", [])])

api_router = APIRouter()
api_router.include_router(rag_router, prefix="/agent", tags=["agent"])

