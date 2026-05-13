from ollama import AsyncClient
from ollama import ChatResponse

# local development
# async function to get llm response from ollama
async def llm_ollama(user_content: str, system_content: str, model: str = "gemma3") -> str | None:
    response: ChatResponse = await AsyncClient().chat(model=model, messages=[
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        ])
    return response.message.content
