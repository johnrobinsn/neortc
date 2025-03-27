
conda


openssl req -x509 -newkey rsa:2048 -keyout mykey.pem -out mycert.pem -days 3650 -nodes


# https://www.electricmonk.nl/log/2018/06/02/ssl-tls-client-certificate-verification-with-python-v3-4-sslcontext/
# https://docs.aiohttp.org/en/v3.8.4/client_advanced.html
# ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
# ssl_context.load_cert_chain('./cert.pem','./key.pem')

# https://stackoverflow.com/questions/72326817/websockets-run-client-side-and-server-side-in-1-script-with-asyncio

    #print('sio connect:',sid,env)
    # if sio.handshake.query.token != token: 
    #     print('Access Denied')
    #     sio.disconnect(True)
leak... 
    https://github.com/aiortc/aiortc/issues/554


    #https://platform.openai.com/docs/api-reference/audio/createSpeech

# curl https://api.openai.com/v1/audio/speech \
#   -H "Authorization: Bearer $OPENAI_API_KEY" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "tts-1",
#     "input": "The quick brown fox jumped over the lazy dog.",
#     "voice": "alloy"
#   }' \
#   --output speech.mp3



        # can add the tracks back to peer connection to reflect back... 


        # else:
#     print('Warning: no matching broadcaster.  Monitoring...')

##############
#OggS
#https://en.wikipedia.org/wiki/Ogg 
#OpusHead
#https://www.opus-codec.org/docs/opusfile_api-0.7/structOpusHead.html

'''
llm is text in... text out... 


llm is prompt history (text, images) in response out

tts is text in... audio (speech out)

stt is audio in... text out... 

could be other audio consumers

could be other video/images consumers

could be a way to acquire audio or video/graphics context

class Agent:
    def __init__()
    def speak():
    def prompt():
    def memory():
    def onMemoryChanged:
    def prompt(msg, flags):
    def processVideoFrame(f):
    def processAudioFrame(f):
    def enableAudioIn(f):
    def enableVideoIn(f):
    def enableVideoOut(f):
    def enableAudioOut(f):
'''

pass current location
pass current date and time

other tools?


# >>> torch.cuda.is_available()
# >>> torch.cuda.get_device_name(torch.cuda.current_device())


#minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
#minds = minds.cast_column("audio", Audio(sampling_rate=16_000))


#split on silence
#https://onkar-patil.medium.com/how-to-remove-silence-from-an-audio-using-python-50fd2c00557d


wrapping socketio decorator in class
https://github.com/miguelgrinberg/python-socketio/issues/390

https://stackoverflow.com/questions/38867763/why-i-have-to-open-data-channel-before-send-peer-connection-offer


alternative ogg streaming...

import logging
import asyncio
import aiohttp
import pyogg
import sounddevice as sd
import opuslib

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class StreamingOpusClient:
    def __init__(self, sample_rate=48000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.decoder = opuslib.Decoder(sample_rate, channels)

    async def fetch_and_play(self, text):
        async with aiohttp.ClientSession() as session:
            async with session.post('https://api.openai.com/v1/tts', json={'text': text}) as response:
                if response.status != 200:
                    log.error(f"Failed to fetch TTS data: {response.status}")
                    return

                ogg_stream = OggStream(self.decoder)
                async for chunk in response.content.iter_chunked(4096):
                    ogg_stream.process_chunk(chunk)

    def process_pcm(self, pcm_data):
        # Play PCM data
        sd.play(pcm_data, samplerate=self.sample_rate, channels=self.channels)

class OggStream:
    def __init__(self, decoder):
        self.decoder = decoder
        self.buffer = b""
        self.pages = []

    def process_chunk(self, chunk):
        self.buffer += chunk
        while True:
            page, self.buffer = self._extract_page(self.buffer)
            if page is None:
                break
            self.pages.append(page)
            self._process_page(page)

    def _extract_page(self, buffer):
        try:
            page = pyogg.OggPage(buffer)
            return page, buffer[page.header_len + page.body_len:]
        except pyogg.OggError:
            return None, buffer

    def _process_page(self, page):
        pcm_data = self.decoder.decode(page.body, frame_size=960)
        sd.play(pcm_data, samplerate=self.decoder.sample_rate, channels=self.decoder.channels)

async def main():
    client = StreamingOpusClient()
    await client.fetch_and_play("Hello, this is a test of the streaming Opus client.")

if __name__ == "__main__":
    asyncio.run(main())