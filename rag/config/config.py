import os
from dataclasses import dataclass


# runtime configuration for the RAG service, loaded once at startup. this is
# the only place in rag/ that reads environment variables; everything else
# receives its settings from a Config
@dataclass(frozen=True)
class Config:
    bucket_name: str
    # one region for every AWS client (S3 and Bedrock)
    region: str = "us-east-1"
    redis_host: str = "localhost"
    redis_port: int = 6379
    # sentence-transformers model used for chunk and query embeddings
    embedding_model_name: str = "all-MiniLM-L6-v2"
    # development LLM provider (llm_ollama)
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3"
    # production LLM provider (llm_bedrock); Bedrock model ids carry the
    # "anthropic." prefix. any Converse-compatible model id works
    bedrock_model_id: str = "anthropic.claude-opus-5-5"


# build a Config from environment variables, falling back to defaults
# for anything optional
def load_config() -> Config:
    bucket_name = os.getenv("S3_BUCKET")
    if not bucket_name:
        raise ValueError("S3_BUCKET must be set")

    port = os.getenv("REDIS_PORT", "6379")
    try:
        redis_port = int(port)
    except ValueError as e:
        raise ValueError(f"invalid REDIS_PORT {port!r}") from e

    return Config(
        bucket_name=bucket_name,
        region=os.getenv("AWS_REGION", "us-east-1"),
        redis_host=os.getenv("REDIS_HOST", "localhost"),
        redis_port=redis_port,
        embedding_model_name=os.getenv(
            "SENTENCE_TRANSFORMERS_MODEL_NAME", "all-MiniLM-L6-v2"
        ),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3"),
        bedrock_model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-opus-5-5"),
    )
