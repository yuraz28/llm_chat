from shemas import Message
from fastapi import APIRouter
from logics import pipeline
from collections import defaultdict


router = APIRouter(
    prefix=""
)

dialogs = defaultdict(list)


@router.post('/message')
async def mes_bot(mes: Message):
    dialogs[f"{mes.user_uuid}:{mes.ws_token}"].append(f'Собеседник: {mes.content}')
    if len(dialogs[f"{mes.user_uuid}:{mes.ws_token}"]) > 7:
        dialogs[f"{mes.user_uuid}:{mes.ws_token}"] = dialogs[f"{mes.user_uuid}:{mes.ws_token}"][2:]
    user = dialogs[f"{mes.user_uuid}:{mes.ws_token}"]
    answer = pipeline(user)
    dialogs[f"{mes.user_uuid}:{mes.ws_token}"].append(f'Ты: {answer}')
    return answer
