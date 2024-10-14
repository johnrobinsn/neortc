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

import logging
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from multiprocessing import Process, Queue
import asyncio

import numpy as np
from transformers import pipeline

from aiortc.mediastreams import MediaStreamError
from samplerate import resample

# from openai_agent.llm_openai import prompt
# from .. import llm_openai.prompt
#from llm_openai import prompt


class STT:
    def __init__(self):
        self.whisper_sample_rate = 16000

        self.capturingAudio=False
        self.audio = []
        self.sample_rate = None
        self.num_channels = None
        self.llm = None
        # self.audioHandler = None

    async def setCaptureAudio(self,f):
        # global capturingAudio
        # global audio
        # global sample_rate
        # global num_channels

        if self.capturingAudio == f:
            return
        self.capturingAudio = f
        print('Capturing Audio:', f)
        # TODO probably set minumum number of frames to something meaningful
        if not self.capturingAudio and len(self.audio) > 0: # on end of capture send to asr
            buffer = np.concatenate(self.audio,axis=1)
            # take the mean across channels to reduce to a single channel
            buffer = buffer.mean(axis=0)
            buffer = buffer[::self.num_channels]
            # buffer.astype(np.int16).tofile('foo48.pcm')

            ratio = self.whisper_sample_rate / self.sample_rate
            buffer = resample(buffer, ratio, 'sinc_best')

            if False:
                buffer.astype(np.int16).tofile('foo.pcm')

                with open('test.pcm', 'wb') as f:
                    f.write(buffer.tobytes())

                print('pcm created... ', buffer.shape, buffer.dtype, ' ', sample_rate)
            
            
            float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max

            t = await self.speech2Text(float_buffer)
            if self.llm:
                await self.llm.prompt(t['text'])
        self.audio = []

    async def speech2Text(self,float_buffer):
        return await w.send('stt',(float_buffer,))
    
    def setLLM(self,llm):
        self.llm = llm

    async def handle_audio(self,track):
        # global capturingAudio
        # global audio
        # global sample_rate
        # global num_channels
        
        # https://stackoverflow.com/questions/31674416/python-realtime-audio-streaming-with-pyaudio-or-something-else
        # isopen = False

        while True:        
            try:
                frame = await track.recv()

                if not self.capturingAudio:
                    continue # drop frame

                self.sample_rate = frame.sample_rate
                self.num_channels = len(frame.layout.channels)
                self.audio.append(frame.to_ndarray())

            except MediaStreamError:
                # This exception is raised when the track ends
                break
        # self.audioHandler = handle_audio()

        # def getAudioHandler(self):
        #     if not self.audioHandler:
        #         self.createAudioHandler()
        #     return self.audioHandler


class Worker:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.inQ = Queue()
        self.outQ = Queue()
        self.p = Process(
            target=self.loop,
            args=(self.inQ, self.outQ))
        self.p.start()

    @staticmethod
    def loop(inQ: Queue, outQ: Queue):
        try:
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
            # asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device=0)
            asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny",device='cpu')
            print('Starting Whisper',asr)

            while True:
                command, args = inQ.get()
                if command == 'stop':
                    break
                elif command == 'stt':
                    outQ.put((asr(args[0])))
        except KeyboardInterrupt:
            pass

    def stop(self):
        self.send('stop')
        self.p.join()
        self.inQ.close()
        self.outQ.close()

    async def send(self,*task):
        async with self.lock:
            self.inQ.put(task)
            return await asyncio.get_running_loop().run_in_executor(None, self.outQ.get)

def startWhisper():
    global w
    w = Worker()

