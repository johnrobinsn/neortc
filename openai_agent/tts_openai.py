# Copyright 2024 John Robinson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file provides a streaming Opus client for OpenAI's TTS webservice.
# Provided a text string to the **say** function it will
# acquire and stream an opus string, will break the stream into
# individual opus segments which can be fed into an audio pipeline
# while the http request is still ongoing.

import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import os
from time import time
from datetime import datetime
import json
from fractions import Fraction
import struct
from asyncio import sleep
import asyncio
from aiohttp import ClientSession
# from av import codec
# from av.packet import Packet
# from aiortc.mediastreams import MediaStreamTrack

from aconfig import config
openai_api_key = config.get('openai_api_key')

class TTS_OpenAI:
    def __init__(self, opus_frame_handler=None):
        self.packetq = []
        self.next_pts = 0
        self.silence_duration = 0.02

        self.time_base = 48000
        self.time_base_fraction = Fraction(1, self.time_base)

        self.gcodec = None
        self.gsample_rate = 0
        self.gchannels = 0
        # self.audioEnabled = False
        # self._createTTSTrack()

        self.text = ''
        # self.muted = False
        self.opus_frame_handler = opus_frame_handler

    # def mute(self, f):
    #     self.muted = f

    # def clearAudio(self):
    #     self.packetq.clear()
    #     self.text = ''
    
    # def enableAudio(self,f):
    #     log.info('enableAudio: %s',f)
    #     if not f:
    #         self.clearAudio()
    #     self.audioEnabled = f
        
    async def say(self,t):
        # TODO do this on the agent side... 
        # if not self.audioEnabled:  
        #     log.info('audio disabled')
        #     return
        # start = datetime.now()
        def on_segment(channels, sample_rate, segment):
            if self.opus_frame_handler:
                self.opus_frame_handler(segment)
        await self._requestTTS(t,on_segment)

    # async def open(self):
    #     self.text = ''

    # async def write(self,text):
    #     self.text = self.text + text
    #     last_newline = max(self.text.rfind('\n'), self.text.rfind('. '))
    #     if last_newline != -1:
    #         await self.say(self.text[:last_newline + 1])
    #         self.text = self.text[last_newline + 1:]
        
    # async def close(self):
    #     if self.text.strip():
    #         await self.say(self.text)
    #     self.text = ''

    # def getTrack(self):
    #     return self.ttsTrack
    
    ## -------------- internal impl --------------
    # TODO move OggProcessor to tts_utils.py
    class _OggProcessor:
        pageMagic = struct.unpack('>I', b'OggS')[0]
        headerMagic = struct.unpack('>Q', b'OpusHead')[0]
        commentMagic = struct.unpack('>Q', b'OpusTags')[0]

        def __init__(self,cb):
            self.cb = cb
            self.buffer = b''
            self.meta = None

        def onMetaPage(self,page,headerSize):
            metaFormat = '<8sBBHIhB'
            metaSize = struct.calcsize(metaFormat)
            (magic, version, channelCount, preSkip, sampleRate, gain, channelMapping) = struct.unpack_from(metaFormat,page,headerSize)

            sampleRate *= 2 # Not sure why we need this... 
            magic = magic.decode('utf-8')

            self.meta = {
                'magic': magic,
                'version': version,
                'channelCount': channelCount,
                'sampleRate': sampleRate,
            }

        def onPage(self,page,headerSize,segmentSizes):
            if self.cb and self.meta: # need the stream metadata
                i = headerSize
                for s in segmentSizes:
                    self.cb(page[i:i+s],self.meta)
                    i = i+s

        # concat buffer and process all available pages
        # if we don't have enough data bail out and wait for more
        def addBuffer(self,b):
            self.buffer = self.buffer + b
            i = 0
            while len(self.buffer) >= i+27:  # enough room for a header
                if self.pageMagic == struct.unpack_from('>I',self.buffer,i)[0]:
                    numSegments = struct.unpack_from('B', self.buffer,i+26)[0]
                    headerSize = 27+numSegments

                    if len(self.buffer) < i+headerSize:
                        return # wait for more data

                    segmentSizes = struct.unpack_from('B'*numSegments,self.buffer,i+27)
                    segmentTotal = sum(segmentSizes)
                    pageSize = headerSize+segmentTotal

                    if len(self.buffer) < i+pageSize:
                        return # wait for more data
                    
                    page = self.buffer[i:i+pageSize]
                    pageDataSize = len(page)-headerSize
                    if pageDataSize >= 8 and self.headerMagic == struct.unpack_from('>Q',page,headerSize)[0]:
                        self.onMetaPage(page,headerSize)
                    elif pageDataSize >= 8 and self.commentMagic == struct.unpack_from('>Q',page,headerSize)[0]:
                        pass # we don't do anything with comment pages
                    else: # Assume audio page
                        self.onPage(page,headerSize,segmentSizes)
                    i = i+pageSize 
                    self.buffer = self.buffer[i:] # done with this page discarding
                    i = 0
                    continue
                i = i + 1

    # invoke OpenAI's TTS API with the provided text(t)
    # and process the returned Opus stream
    async def _requestTTS(self, t, callback):
        url = 'https://api.openai.com/v1/audio/speech'

        headers = {
            'Authorization': f'Bearer {openai_api_key}',
            'Content-Type': 'application/json',
        }

        data = {
            'model': 'tts-1',
            'input': t,
            'voice': 'echo', #'alloy',
            'response_format': 'opus',
            'speed': 1.0
        }

        oggFileData = bytearray()

        async with ClientSession() as session:
            async with session.post(url=url,json=data,headers=headers,chunked=True) as response:

                def new_path(segment, meta):
                    callback(meta['channelCount'],meta['sampleRate'],segment)

                oggProcessor = TTS_OpenAI._OggProcessor(new_path)
                if response.status != 200:
                    log.error('OpenAI TTS Call Failed Status:', response.status)
                async for data in response.content.iter_chunked(16384):
                    oggProcessor.addBuffer(data)
                    oggFileData.extend(data)

                dir = f'dictations/{datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")}'
                os.makedirs(dir, exist_ok=True)
                metadata = {'text': t, 
                            'model': 'tts-1', 'voice': 'echo', 'speed': 1.0, 
                            'timestamp': datetime.utcnow().isoformat()
                        }
                metadata_path = os.path.join(dir, 'metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                oggfile_path = os.path.join(dir, 'output.ogg')
                with open(oggfile_path, 'wb') as f:
                    f.write(oggFileData)

## -------------- main --------------

def main():
    import aiofiles
    import sys
    from termcolor import colored

    # load_dotenv(".env.local")

    from tts_utils import OpusStreamPlayer

    opus_player = OpusStreamPlayer()

    def opus_frame_handler(opus_frame):
        opus_player.write(bytearray(opus_frame)) # TODO should byte array be pushed down to the player?

    print(colored(f'TTS_OpenAI', 'cyan'))

    tts = TTS_OpenAI(opus_frame_handler=opus_frame_handler)

    async def read_lines():

        await tts.say('The sky above the port was the color of television, tuned to a dead channel.')
        await tts.say('hello world hello world hello world')    

        async with aiofiles.open('/dev/stdin', mode='r') as f:
            print(colored("> ","yellow"), end="")
            sys.stdout.flush()
            async for line in f:    
                # print(colored(await askit.prompt1(line.strip(), moreTools=[get_current_location]),"green"))
                await tts.say(line.strip())
                print(colored("> ","yellow"), end="")
                sys.stdout.flush()

    asyncio.run(read_lines())

if __name__ == '__main__':
    main()