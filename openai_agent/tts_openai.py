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
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import struct
from time import time
from datetime import datetime
from fractions import Fraction
from asyncio import sleep
from aiohttp import ClientSession
from av import codec
from av.packet import Packet
from aiortc.mediastreams import MediaStreamTrack

from api_key_openai import api_key

packetq = []
next_pts = 0
silence_duration = 0.02

time_base = 48000
time_base_fraction = Fraction(1, time_base)

gcodec = None
gsample_rate = 0
gchannels = 0

class OggProcessor:
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
                if self.headerMagic == struct.unpack_from('>Q',page,headerSize)[0]:
                    self.onMetaPage(page,headerSize)
                elif self.commentMagic == struct.unpack_from('>Q',page,headerSize)[0]:
                    pass # we don't do anything with comment pages
                else: # Assume audio page
                    self.onPage(page,headerSize,segmentSizes)
                i = i+pageSize 
                self.buffer = self.buffer[i:] # done with this page discarding
                i = 0
                continue
            i = i + 1

def get_silence_packet(duration_seconds):
    global next_pts
    chunk = bytes.fromhex('f8 ff fe')

    pkt = Packet(chunk)
    pkt.pts = next_pts
    pkt.dts = next_pts
    pkt.time_base = time_base_fraction

    pts_count = round(duration_seconds * time_base)
    next_pts += pts_count

    return pkt

# if we we have audio queued deliver that; otherwise silence
def get_audio_packet():
    global packetq
    global next_pts
    global silence_duration
    if len(packetq) > 0:
        try:
            duration,pts_count,chunk = packetq.pop()

            pkt = Packet(chunk)
            pkt.pts = next_pts
            pkt.dts = next_pts
            pkt.time_base = time_base_fraction

            next_pts += pts_count

            return pkt,duration
        except:
            pass # Ignore Empty exception

    return get_silence_packet(silence_duration), silence_duration    

class tts_track(MediaStreamTrack):
    kind = "audio"
    
    def __init__(self):
        super().__init__()
        self.stream_time = None
        log.info('create tts_track')

    async def close(self):
        super().stop()

    async def recv(self):
        try: # exceptions that happen here are eaten... so log them
            packet, duration = get_audio_packet()

            if self.stream_time is None:
                self.stream_time = time()

            wait = self.stream_time - time()
            await sleep(wait)

            self.stream_time += duration
            return packet
        except Exception as e:
            log.error('Exception:', e)
            raise

# invoke OpenAI's TTS API with the provided text(t)
# and process the returned Opus stream
async def requestTTS(t, callback):
    url = 'https://api.openai.com/v1/audio/speech'

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'model': 'tts-1',
        'input': t,
        'voice': 'echo', #'alloy',
        'response_format': 'opus',
        'speed': 1.0
    }

    async with ClientSession() as session:
        async with session.post(url=url,json=data,headers=headers,chunked=True) as response:

            def new_path(segment, meta):
                callback(meta['channelCount'],meta['sampleRate'],segment)

            oggProcessor = OggProcessor(new_path)
            if response.status != 200:
                log.error('OpenAI TTS Call Failed Status:', response.status)
            async for data in response.content.iter_chunked(16384):
                oggProcessor.addBuffer(data)                                   

def init_codec(channels, sample_rate):
    global gcodec
    global gsample_rate
    global gchannels

    gcodec = codec.CodecContext.create('opus', 'r')

    gcodec.sample_rate = sample_rate
    gcodec.channels = channels

    gsample_rate = sample_rate
    gchannels = channels

async def say(t):
    start = datetime.now()
    def on_segment(channels, sample_rate, segment):
        global gcodec
        global gsample_rate
        global gchannels
        global packetq
        nonlocal start
        if start:
            log.info('time to first segment:', (datetime.now()-start).total_seconds())
            start = None

        if gsample_rate != sample_rate or gchannels != channels:
            init_codec(channels, sample_rate)

        sample_count = 0
        for frame in gcodec.decode(Packet(segment)):
            sample_count += frame.samples

        duration = sample_count / gsample_rate

        pts_count = round(duration * time_base)

        packetq.insert(0,(duration, pts_count, segment))
    await requestTTS(t,on_segment)
