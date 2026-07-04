from pydantic import BaseModel
from typing import Optional

class RagRequest(BaseModel):
    query: str


class RAGUsedContext(BaseModel):
    image_url: str
    price: Optional[float] = None
    description: str

class RagResponse(BaseModel):
    answer: str
    used_context: list[RAGUsedContext]
     