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
import torch
from transformers import pipeline

from aiortc.mediastreams import MediaStreamError
from samplerate import resample
import wave
from pydub import AudioSegment

# from openai_agent.llm_openai import prompt
# from .. import llm_openai.prompt
#from llm_openai import prompt


class STT:
    def __init__(self):
        # TODO can this be slimmed down... what's actually used?
        self.whisper_sample_rate = 16000

        self.capturingAudio=False
        self.audio = []
        self.sample_rate = None
        self.num_channels = None
        self.llm = None
        self.audioBuffer = []
        self.id = STT._nextId
        STT._nextId += 1
        STT._instances.append(self)

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

    async def setCaptureAudio(self,f):

        if self.capturingAudio == f:
            return
        self.capturingAudio = f
        print('Capturing Audio:', f)
        # TODO probably set minumum number of frames to something meaningful
        # if not self.capturingAudio and len(self.audio) > 0: # on end of capture send to asr
        #     #TODO should I empty self.audio when starting capture.
        #     buffer = np.concatenate(self.audio,axis=1)
        #     # take the mean across channels to reduce to a single channel
        #     print('buffer2:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
        #     buffer = buffer.mean(axis=0)
        #     print('buffer3:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
        #     buffer = buffer[::self.num_channels]
        #     print('buffer4:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
        #     # buffer.astype(np.int16).tofile('foo48.pcm')

        #     ratio = self.whisper_sample_rate / self.sample_rate
        #     buffer = resample(buffer, ratio, 'sinc_best')

        #     if False:
        #         buffer.astype(np.int16).tofile('foo.pcm')

        #         with open('test.pcm', 'wb') as f:
        #             f.write(buffer.tobytes())

        #         print('pcm created... ', buffer.shape, buffer.dtype, ' ', sample_rate)
            
            
        #     # implement a high pass filter for the audio data in buffer



        #     float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max

        #     t = await self.speech2Text(float_buffer)
        #     if self.llm:
        #         await self.llm.prompt(t['text'])
        # self.audio = []

    # async def speech2Text(self,float_buffer):
    #     return await w.send('stt',(float_buffer,))
    
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
                f = frame.to_ndarray()
                # print('frame:', f.shape, f.dtype, ' ', frame.sample_rate)
                # print('frame:', f.min(), f.max(), f.mean(), f.std())
                # print('channels:', frame.layout.channels)
                self.audio.append(f)

                if len(self.audio) >= 8: #16:
                    buffer = np.concatenate(self.audio, axis=1)
                    # print('buffer:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
                    # processed_buffer = await self.processor.process_audio(buffer, self.sample_rate)
                    # await w.send('stt',(buffer,))
                    w.inQ.put(('process',(buffer,self.id)))
                    self.audio = []  # Clear the buffer after processing

            except MediaStreamError:
                # This exception is raised when the track ends
                break
        # self.audioHandler = handle_audio()

        # def getAudioHandler(self):
        #     if not self.audioHandler:
        #         self.createAudioHandler()
        #     return self.audioHandler


async def process_out_queue(outQ: Queue):
    loop = asyncio.get_event_loop()
    while True:
        print('waiting for outq result')
        command,result,stt = await loop.run_in_executor(None, outQ.get)
        print('got outq result')
        if command == 'stop':
            break
        print("Processed result:", result,stt)
        s = STT.getInstance(stt)
        if s:
            if result:
                await s.llm.prompt(result['text'])
            else:
                await s.llm.bargeIn()
        
        # Process the result here

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
            asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny",device='cpu')
            print('Starting Whisper',asr)
            buffers = {} # TODO should this be a class variable?
            bargeIn = False
            silenceCount = 0
            bargeInCount = 0

            def process(args):
                # nonlocal bargeIn
                nonlocal silenceCount
                nonlocal bargeInCount
                buffer, stt = args
                if stt not in buffers:
                    buffers[stt] = []
                # print('buffer:', buffer.shape, buffer.dtype)
                buffer = buffer.mean(axis=0)
                # buffer = buffer[::self.num_channels]
                buffer = buffer[::2]
                # print('buffer mono:', buffer.shape, buffer.dtype)
                # ratio = self.whisper_sample_rate / self.sample_rate
                # ratio = 16000 / 48000
                #TODO this is a hack to get the ratio right
                ratio = 16000 / 48000

                buffer.astype(np.int16).tofile('foo48.pcm')
                buffer = resample(buffer, ratio, 'sinc_best')
                buffer = buffer.astype(np.int16)
                print('resampled buffer:', buffer.shape, buffer.dtype)

                float_buffer2 = buffer.astype(np.float32) / np.iinfo(np.int16).max
                # print('float_buffer2:', float_buffer2.shape, float_buffer2.dtype)
                float_buffer2 = float_buffer2.reshape(-1,512)
                # print('float_buffer2a:', float_buffer2.shape, float_buffer2.dtype)
                p = model(torch.from_numpy(float_buffer2),16000)
                # print('VAD:', p.shape, p)
                p = torch.all(p<0.25)
                # print('VAD2:', p)


                # Silence detection
                rms = np.sqrt(np.mean(buffer**2))
                # silence_threshold = 600  # Adjust this threshold as needed
                silence_threshold = 50  # Adjust this threshold as needed
                # print('rms:', rms)

                is_silent = rms < silence_threshold
                # is_silent = False

                if not is_silent:
                    is_silent = p # noisy but no voice detected

                if is_silent:
                    bargeInCount = 0
                    # buffers[stt] = []
                    silenceCount = silenceCount + 1
                else:
                    bargeInCount = bargeInCount + 1
                    # if silenceCount > 0:
                    #     buffers[stt] = []
                    silenceCount = 0

                if silenceCount >= 2:
                    # print('Silence detected, skipping processing')
                    if len(buffers[stt]) < 2:
                        buffers[stt] = []
                        # bargeIn = False
                        return
                    # print('Processing buffered audio', len(Worker.bbuffer))
                    buffer = np.concatenate(buffers[stt], axis=0)
                    print('buffer2:', buffer.shape, buffer.dtype)
                    float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
                    # print('float_buffer:', float_buffer.shape, float_buffer.dtype)

                    # create file name with date and time
                    import datetime
                    now = datetime.datetime.now()
                    filename = now.strftime("%Y%m%d-%H%M%S")
                    # filename = filename + '.pcm'
                    # with wave.open(filename, 'wb') as f:
                    #     f.setnchannels(1)
                    #     f.setsampwidth(2)
                    #     f.setframerate(16000)
                    #     f.writeframes(buffer.tobytes())

                    # buffer.astype(np.int16).tofile('foo.pcm')

                    filename = filename + '.mp3'
                    audio_segment = AudioSegment(
                        buffer.tobytes(), 
                        frame_rate=16000,
                        sample_width=2,  # 2 bytes for 16-bit audio
                        channels=1
                    )

                    # Export the AudioSegment to an MP3 file
                    audio_segment.export(filename, format="mp3")

                    t = asr(float_buffer)
                    print('stt:', t)
                    outQ.put(('process',t,stt))
                    buffers[stt] = []
                    bargeIn = False
                    # if stt.llm:
                    #     asyncio.get_event_loop().create_task(stt.llm.prompt(t['text']))
                    
                else:
                    buffers[stt].append(buffer)
                    if bargeInCount == 2: # Only send bargeIn once
                        print('BargeIn detected')
                        outQ.put(('process',None,stt))

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

