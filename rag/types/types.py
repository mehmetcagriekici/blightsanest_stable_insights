from pydantic import BaseModel

# default document model before building the microservices
class Document(BaseModel):
    id: str
    content: str

# RAG response
class RagResponse(BaseModel):
    id: str
    status: str
    response: str
