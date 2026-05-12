from pydantic import BaseModel

# default document model before building the microservices
class Document(BaseModel):
    id: str
    content: str
