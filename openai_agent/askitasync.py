import askit
from dotenv import load_dotenv

load_dotenv('.env.local')

async def switch_lights(light_name: str, state: bool):
    """
    Switches the lights on or off given the light name

    Returns:
        bool: True if the lights were switched on or off successfully, False otherwise

    """
    print('Switching lights: ', light_name, state)
    return True


ask = askit.AskIt()

async def prompt(messages):
    async def gatherStrings(stream):
        s = ""
        async for chunk in stream:
            s = s + chunk
        return s
    return await gatherStrings(ask.streamPrompt(messages))

# streaming version
async def streamPrompt(messages):
    stream = ask.streamPrompt(messages, moreTools=[switch_lights])
    async for chunk in stream:
        yield chunk

def startWorker():
    # Nothing to do
    pass