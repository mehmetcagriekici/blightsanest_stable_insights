from pydantic import BaseModel

# default document model before building the microservices
class Document(BaseModel):
    id: str
    content: str

# RAG response
class RagResponse(BaseModel):
    # the LLM only returns status + response; id is optional and may be
    # populated server-side when a response needs to be tied to a record
    id: str | None = None
    status: str
    response: str

# default user for development 
class User(BaseModel):
    id: str
    aws_access_key_id: str
    aws_secret_access_key: str
    region: str
    bucket_name: str
