from uuid import UUID
from msgpack import Timestamp
from pydantic import BaseModel

# database types
# user
class DbUser(BaseModel):
    id: UUID
    username: str
    email: str
    hashed_password: str
    created_at: Timestamp
    updated_at: Timestamp

# document
class DbDocument(BaseModel):
    id: UUID
    user_id: UUID
    created_at: Timestamp
    updated_at: Timestamp
