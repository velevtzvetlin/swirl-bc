from pydantic import BaseModel

class RagRequest(BaseModel):
    query: str

class RagResponse(BaseModel):
    answer: str