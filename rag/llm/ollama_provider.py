import logging

import httpx
from ollama import AsyncClient, ChatResponse, ResponseError

logger = logging.getLogger(__name__)


# local development / containerized
# async function to get llm response from ollama
#
# host and model come from Config.ollama_host / Config.ollama_model
# (OLLAMA_HOST, http://ollama:11434 when running in Docker; OLLAMA_MODEL).
# bind them once at startup so RAG gets the plain (user, system) signature:
#     functools.partial(
#         llm_ollama, host=config.ollama_host, model=config.ollama_model
#     )
async def llm_ollama(
    user_content: str, system_content: str, *, host: str, model: str
) -> str | None:
    try:
        response: ChatResponse = await AsyncClient(host=host).chat(
            model=model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )
        return response.message.content
    except (ResponseError, ConnectionError, httpx.HTTPError) as e:
        logger.error("ollama chat call failed: %s", e)
        return None
