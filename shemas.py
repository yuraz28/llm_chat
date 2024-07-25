from pydantic import BaseModel
from uuid import UUID


class Message(BaseModel):
    user_uuid: UUID
    ws_token: str
    content: str
