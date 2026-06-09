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

# default user
class User(BaseModel):
    id: str
    aws_access_key_id: str
    aws_secret_access_key: str
    region: str
    bucket_name: str
