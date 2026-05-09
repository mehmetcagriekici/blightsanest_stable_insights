from pydantic import BaseModel

class Document(BaseModel):
    id: str
    content: str
