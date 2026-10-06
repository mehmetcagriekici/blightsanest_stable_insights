import asyncio
import logging

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


# production / containerized
# async function to get an llm response from aws bedrock via the converse api
# (the converse api is model-agnostic - same request/response shape for any model)
#
# region and model_id come from Config (region, bedrock_model_id). bind them
# once at startup so RAG gets the plain (user, system) signature:
#     functools.partial(llm_bedrock, region=..., model_id=...)
# model_id is the model-agnostic knob: any Converse-compatible model works
async def llm_bedrock(
    user_content: str, system_content: str, *, region: str, model_id: str
) -> str | None:
    try:
        # client creation raises NoRegionError (a BotoCoreError) when no region
        # is configured, so it belongs inside the try as well
        client = boto3.client("bedrock-runtime", region_name=region)
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
        logger.error("bedrock converse call failed: %s", e)
        return None

    # return None on a malformed response so RAG.rag's None guard can handle it
    try:
        return response["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error("unexpected bedrock response shape: %s", e)
        return None
