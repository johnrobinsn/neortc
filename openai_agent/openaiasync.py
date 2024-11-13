import asyncio
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
from datetime import datetime

from openai import AsyncOpenAI
# from auth_openai import api_key
from aconfig import config
import uuid

openai_api_key = config.get('openai_api_key')

client = AsyncOpenAI(api_key=openai_api_key)

async def prompt(messages):
    start = datetime.now()
    response = await client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
    )
    log.info('Time for LLM Response: %d', (datetime.now()-start).total_seconds())
    r = response.choices[0].message.content
    print('openai response:', r)
    return r

def startWorker():
    # Nothing to do
    pass