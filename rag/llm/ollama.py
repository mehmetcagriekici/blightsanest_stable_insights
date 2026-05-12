from ollama import AsyncClient
from ollama import ChatResponse

# local development
# async function to get llm response from ollama
async def llm_ollama(content: str, role: str = "user", model: str = "gemma3"):
    response: ChatResponse = await AsyncClient().chat(model=model, messages=[{"role": role, "content": content}])
    return response.message.content
