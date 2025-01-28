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

import os
import datetime
import logging
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from multiprocessing import Process, Queue
import asyncio

import numpy as np
import torch
from transformers import pipeline

from aiortc.mediastreams import MediaStreamError
from samplerate import resample
import wave
from pydub import AudioSegment

# STT state
# initializing, ready, error

# STT Event
# onStateChange, onTranscript, onVoiceDetected onVoiceNotDetected

from typing import Literal

STTEventType = Literal[
    "voice_detected",
    "voice_not_detected",
    "final_transcript",
    "state_change",
]

class STT:
    def __init__(self):
        # TODO can this be slimmed down... what's actually used?
        self.whisper_sample_rate = 16000

        self.capturingAudio=False
        self.audio = []
        self.fullCapture = []
        self.sample_rate = None
        self.num_channels = None
        self.audioBuffer = []
        self.id = STT._nextId
        STT._nextId += 1
        STT._instances.append(self)
        self.captureDir = None
        self.listeners = []

    def __del__(self):
        STT._instances.remove(self)

    _nextId = 0
    _instances = []

    @staticmethod
    def getInstance(id):
        for i in STT._instances:
            if i.id == id:
                return i
        return None
    
    def addListener(self,listener):
        self.listeners.append(listener)

    def removeListener(self,listener):
        self.listeners.remove(listener)

    def notifyListeners(self,eventType,data):
        for l in self.listeners:
            l(self,eventType,data)

    async def setCaptureAudio(self,f):

        if self.capturingAudio == f:
            return
        self.capturingAudio = f
        print('Capturing Audio:', f)

        if self.capturingAudio:
            self.audio = []
            self.fullCapture = []
            now = datetime.datetime.now()
            dirname = f'recordings/{now.strftime("%Y%m%d-%H%M%S")}'
            if not os.path.exists(dirname):
                os.makedirs(dirname,exist_ok=True)
            self.captureDir = dirname


        if not self.capturingAudio and len(self.fullCapture) > 1:
            buffer = np.concatenate(self.fullCapture, axis=1)

            filename = f'{self.captureDir}/fullcapture.mp3'
            audio_segment = AudioSegment(
                buffer.tobytes(), 
                frame_rate=self.sample_rate,#16000,
                sample_width=2,  # 2 bytes for 16-bit audio
                channels=self.num_channels#1
            )

            # Export the AudioSegment to an MP3 file
            audio_segment.export(filename, format="mp3")
            self.fullCapture = []

        # TODO probably set minumum number of frames to something meaningful

    async def handle_audio(self,track):
        # https://stackoverflow.com/questions/31674416/python-realtime-audio-streaming-with-pyaudio-or-something-else

        while True:        
            try:
                frame = await track.recv()

                if not self.capturingAudio:
                    continue # drop frame

                self.sample_rate = frame.sample_rate
                self.num_channels = len(frame.layout.channels)
                f = frame.to_ndarray()
                # print('frame:', f.shape, f.dtype, ' ', frame.sample_rate)
                # print('frame:', f.min(), f.max(), f.mean(), f.std())
                # print('channels:', frame.layout.channels)
                self.fullCapture.append(f)
                self.audio.append(f)

                if len(self.audio) >= 8: # 8=>320ms, 16=>640ms
                    buffer = np.concatenate(self.audio, axis=1)
                    # buffer: (1, 15360) int16   48000 with 8 frames
                    # print('buffer:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
                    # processed_buffer = await self.processor.process_audio(buffer, self.sample_rate)
                    # await w.send('stt',(buffer,))
                    w.inQ.put(('process',(buffer,self.id,self.captureDir)))
                    self.audio = []  # Clear the buffer after processing

            except MediaStreamError:
                # This exception is raised when the track ends
                break



async def process_out_queue(outQ: Queue):
    loop = asyncio.get_event_loop()
    while True:
        command,result,stt = await loop.run_in_executor(None, outQ.get)
        if command == 'stop':
            break
        s = STT.getInstance(stt)
        if s:
            s.notifyListeners(command,result)
        else:
            print('STT instance not found')        


# executor = ThreadPoolExecutor(1)

# async def search_internet(query: str) -> str:
#     """
#     Searches the internet for the given query and returns the top search results.

#     Args:
#         query (str): The search query.

#     Returns:
#         str: The top search results.
#     """
#     loop = asyncio.get_event_loop()
#     results = await loop.run_in_executor(executor, DDGS().text, query, 5)

segment_state = [
    "init",
    "voice detected",
    "selence 1",
    "silence 2",
    "silence 3",
    "voice not detected"
]

class Worker:
    def __init__(self):
        self.inQ = Queue()
        self.outQ = Queue()
        self.p = Process(
            target=self.loop,
            args=(self.inQ, self.outQ))
        self.p.start()

    @staticmethod
    def loop(inQ: Queue, outQ: Queue):
        try:
            print('Loading Silero VAD')
            model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=True)
            print('Loaded Silero VAD')

            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=0)
            asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device='cpu')
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-medium.en",device='cpu')
            print('Starting Whisper',asr)
            buffers = {} # TODO should this be a class variable?
            runs = {}
            bargeIn = False
            silenceCount = 0
            bargeInCount = 0

            def process(args):
                # nonlocal bargeIn
                nonlocal silenceCount
                # nonlocal bargeInCount
                buffer, stt, captureDir = args
                if stt not in buffers:
                    buffers[stt] = []
                if stt not in runs:
                    runs[stt] = []
                # print('buffer:', buffer.shape, buffer.dtype, stt)
                buffer = buffer.mean(axis=0) #warning implicit float conversion
                # buffer = buffer[::self.num_channels]
                buffer = buffer[::2]  # convert to mono but dropping a channel
                # buffer = ((buffer[::2] + buffer[1::2])/2).astype(np.int16)  # convert to mono by averaging channels
                # print('buffer mono:', buffer.shape, buffer.dtype)
                # ratio = self.whisper_sample_rate / self.sample_rate
                # ratio = 16000 / 48000
                #TODO this is a hack to get the ratio right
                ratio = 16000 / 48000

                # buffer.astype(np.int16).tofile('foo48.pcm')
                buffer = resample(buffer, ratio, 'sinc_best')
                buffer = buffer.astype(np.int16)
                # print('resampled buffer:', buffer.shape, buffer.dtype)

                float_buffer2 = buffer.astype(np.float32) / np.iinfo(np.int16).max
                # print('float_buffer2:', float_buffer2.shape, float_buffer2.dtype)
                float_buffer3 = float_buffer2
                float_buffer2 = float_buffer2.reshape(-1,512)
                # print('float_buffer2a:', float_buffer2.shape, float_buffer2.dtype)
                p = model(torch.from_numpy(float_buffer2),16000)
                # print('VAD:', p.shape, p)
                p = torch.all(p<0.25)
                # print('VAD2:', p)


                # Silence detection
                rms = np.sqrt(np.mean(buffer**2))
                # silence_threshold = 600  # Adjust this threshold as needed
                silence_threshold = 5  # Adjust this threshold as needed
                # print('rms:', rms)

                # # Silence detection
                # rms = np.sqrt(np.mean(float_buffer3**2))
                # print('rms:', rms)
                # # silence_threshold = 600  # Adjust this threshold as needed
                # silence_threshold = 0.001  # Adjust this threshold as needed
                # # print('rms:', rms)                
                

                is_silent = rms < silence_threshold
                # is_silent = False

                if not is_silent:
                    is_silent = p # noisy but no voice detected

                if not is_silent:
                    silenceCount = 0
                    runs[stt].append(buffer)
                else:
                    silenceCount = silenceCount + 1
                    if silenceCount >= 4 and len(runs[stt]) >= 2:
                        buffers[stt].extend(runs[stt])
                        # print('Buffering audio', len(buffers[stt]))
                        # if bargeInCount == 1: # Only send bargeIn once
                        # print('BargeIn detected')
                        # outQ.put(('process',None,stt))   # voice detected                      
                        outQ.put(('voice_detected',None,stt))   # voice detected
                        runs[stt] = []

                if silenceCount >= 4 and len(buffers[stt]) >= 2:
                    # print('Silence detected, skipping processing')
                    # print('Processing buffered audio', len(Worker.bbuffer))
                    buffer = np.concatenate(buffers[stt], axis=0)
                    # print('buffer2:', buffer.shape, buffer.dtype)
                    float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
                    # print('float_buffer:', float_buffer.shape, float_buffer.dtype)

                    # create file name with date and time
                    
                    now = datetime.datetime.now()
                    # filename = 
                    # if not os.path.exists('recordings'):
                    #     os.makedirs('recordings')

                    filename = f'{captureDir}/{now.strftime("%Y%m%d-%H%M%S")}.mp3'
                    audio_segment = AudioSegment(
                        buffer.tobytes(), 
                        frame_rate=16000,
                        sample_width=2,  # 2 bytes for 16-bit audio
                        channels=1
                    )

                    # Export the AudioSegment to an MP3 file
                    audio_segment.export(filename, format="mp3")

                    t = asr(float_buffer)
                    # print('stt:', t)
                    # outQ.put(('process',t,stt))
                    outQ.put(('final_transcript',t['text'],stt))
                    outQ.put(('voice_not_detected',None,stt))   # voice not detected
                    buffers[stt] = []
                    # bargeIn = False
                    # if stt.llm:
                    #     asyncio.get_event_loop().create_task(stt.llm.prompt(t['text']))
                    
                # else:
                #     buffers[stt].append(buffer)
                #     print('Buffering audio', len(buffers[stt]))
                #     if bargeInCount == 1: # Only send bargeIn once
                #         print('BargeIn detected')
                #         outQ.put(('process',None,stt))

            while True:
                command, args = inQ.get()
                if command == 'stop':
                    break
                elif command == 'process':
                    process(args)
                # elif command == 'stt':
                #     outQ.put((asr(args[0])))
        except KeyboardInterrupt: # TODO does this get hit
            pass
        log.info('Exiting Whisper Loop')

    async def stop(self):
        self.inQ.put(('stop',''))
        self.p.join()
        self.inQ.close()
        self.outQ.close()

# TODO could make this async?
def startWhisper():
    global w
    w = Worker()
    asyncio.get_event_loop().create_task(process_out_queue(w.outQ))

async def stopWhisper():
    global w
    if w:
        log.info('Stopping Whisper')
        await asyncio.get_event_loop().run_in_executor(None, w.outQ.put, ('stop', '', ''))
        await w.stop()
        w = None


if __name__ == "__main__":

    stt = STT()

    stt.addListener(lambda s,e,d: print(f"STT Event: {e} {d}"))

    startWhisper()

    # Example of loading an MP3 file with pydub
    audio_file = "/home/jr/Downloads/neortc/openai_agent/recordings/20250124-161641/fullcapture.mp3"
    audio_segment = AudioSegment.from_mp3(audio_file)

    # Print some information about the audio file
    print(f"Duration: {audio_segment.duration_seconds} seconds")
    print(f"Channels: {audio_segment.channels}")
    print(f"Frame rate: {audio_segment.frame_rate} Hz")

    # Get the raw audio data as a bytestring
    raw_data = audio_segment.raw_data

    # Convert the raw data to a numpy array
    buffer = np.frombuffer(raw_data, dtype=np.int16)

    # Print some information about the buffer
    print(f"Buffer shape: {buffer.shape}")
    print(f"Buffer dtype: {buffer.dtype}")

    # buffer: (1, 15360) int16   48000

    chunksize = 15360

    for c in range(0,buffer.shape[0],chunksize):
        chunk = buffer[c:c+chunksize]
        if chunk.shape[0] < chunksize:
            chunk = np.pad(chunk, (0, chunksize - chunk.shape[0]), 'constant')
        chunk = chunk.reshape(1,chunksize)
        # print(f"Chunk shape: {chunk.shape}")
        # await w.send('process',(chunk,0,'test/blah'))
        w.inQ.put(('process',(chunk,stt.id,'test/blah')))

    # asyncio.get_event_loop().run_until_complete(stopWhisper())
    # asyncio.get_event_loop().run_forever()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    # asyncio
    # await stopWhisper()
    print('Done')