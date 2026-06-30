# blightsanest RAG
from collections.abc import Awaitable, Callable
from helpers.helpers import parse_json
from custom_types.custom_types import Document, RagResponse


class RAG:
    SYSTEM_PROMPT = """
    You are the intelligent assistant of the BlightSanest RAG System.

    BlightSanest - an anagram from the stable insights - is an application where users store and query their own documents.

    Examples:
    Documents might be medical records of a user, and the user might search for patterns in their symptoms.
    Documents might the pages of a journal, and the user might want to learn, what they were busy with a
    month ago.

    Your sole purpose is to answer the user's question using only the retrieved documents provided in the user prompt. 
    Do not use outside knowledge. If the answer cannot be found in the retrieved documents, say so clearly, changing the response status to not found.

    Rules:
        - Only use information from the retrieved documents
        - Do not make assumptions or fill gaps with outside knowledge
        - Be concise and direct
        - If documents are insufficient, say "I don't have enough information in your documents to answer this"

    Respond only in JSON format:
        {
            "status": "found" | "not found",
            "response": "your answer here"
        }
    No extra text, no markdown, no backticks.
    """

    USER_PROMPT_TEMPLATE = """
    Query: {query}

    Retrieved Documents:
        {retrieved_documents}

    Answer the query using only the documents above.
    """

    def __init__(self, generate: Callable[[str, str], Awaitable[str]]) -> None:
        self.generate = generate

    async def rag(self, query: str, retrieved_documents: list[Document]) -> RagResponse:
        # format documents as readable, citable text instead of leaking the
        # pydantic list repr into the prompt
        formatted_documents = "\n\n".join(
            f"[{document.id}] {document.content}" for document in retrieved_documents
        )
        user_prompt = self.USER_PROMPT_TEMPLATE.format(query=query, retrieved_documents=formatted_documents)
        response = await self.generate(user_prompt, self.SYSTEM_PROMPT)
        if response is None:
            raise ValueError("llm did not produce any response")

        data = parse_json(response)
        if "status" not in data or "response" not in data:
            raise ValueError("invalid llm response")

        return RagResponse(**data)


    















