import asyncio
import os

import boto3
from botocore.client import logging
from botocore.exceptions import BotoCoreError, ClientError

# model id is the model-agnostic knob: any Converse-compatible model works
region_name = os.getenv('AWS_REGION_NAME', 'us-east-1')
model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')

# production / containerized
# async function to get an llm response from aws bedrock via the converse api
# (the converse api is model-agnostic - same request/response shape for any model)
async def llm_bedrock(user_content: str, system_content: str) -> str | None:
    client = boto3.client("bedrock-runtime", region_name=region_name)

    try:
        # boto3 is synchronous, so run the call in a thread to avoid blocking
        # the event loop. the system prompt is a top-level parameter, not a
        # message role (converse messages only allow user/assistant)
        response = await asyncio.to_thread(
            client.converse,
            modelId=model_id,
            messages=[
                {"role": "user", "content": [{"text": user_content}]},
            ],
            system=[{"text": system_content}],
        )
    except (BotoCoreError, ClientError) as e:
        logging.error("bedrock converse call failed: %s", e)
        return None

    # return None on a malformed response so RAG.rag's None guard can handle it
    try:
        return response["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError) as e:
        logging.error("unexpected bedrock response shape: %s", e)
        return None
