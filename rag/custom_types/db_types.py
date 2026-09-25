from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


# database types
# user
class DbUser(BaseModel):
    id: UUID
    username: str
    email: str
    hashed_password: str
    created_at: datetime
    updated_at: datetime


# document
class DbDocument(BaseModel):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
