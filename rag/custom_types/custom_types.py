from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    id: str
    content: str


# RAG response
class RagResponse(BaseModel):
    # the LLM only returns status + response; id is optional and may be
    # populated server-side when a response needs to be tied to a record
    id: str | None = None
    status: Literal["found", "not found"]
    response: str


class User(BaseModel):
    id: str
