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

from time import time
import numpy as np

from aiortc.mediastreams import MediaStreamError
from openai_agent.llm_openai import prompt
from transformers import pipeline
from samplerate import resample

whisper_sample_rate = 16000

capturingAudio=False
audio = []
sample_rate = None
num_channels = None

async def setCaptureAudio(f):
    global capturingAudio
    global audio
    global sample_rate
    global num_channels

    if capturingAudio == f:
        return
    capturingAudio = f
    #print('Capturing Audio:', f)
    # todo probably set minumum number of frames to something meaningful
    if not capturingAudio and len(audio) > 0: # on end of capture send to asr
        start_time = time.time()

        #if self.recording:
        #sample_rate = frame.sample_rate
        #num_channels = len(frame.layout.channels)

        #audio_frames.append(frame.to_ndarray())
        #print('num channels:',num_channels)
        #print('audio len:', len(audio))
        #print('audio:',audio[0])
        buffer = np.concatenate(audio,axis=1)
        #print('shape1:', buffer.shape)
        #print('buffer shape:', buffer.shape)
        # take the mean across channels to reduce to a single channel
        buffer = buffer.mean(axis=0)
        buffer = buffer[::num_channels]
        #print('shape2:', buffer.shape)
        #print("old:", len(buffer))
        buffer.astype(np.int16).tofile('foo48.pcm')

        ratio = whisper_sample_rate / sample_rate
        buffer = resample(buffer, ratio, 'sinc_best')

        if False:
            buffer.astype(np.int16).tofile('foo.pcm')

            with open('test.pcm', 'wb') as f:
                f.write(buffer.tobytes())

            print('pcm created... ', buffer.shape, buffer.dtype, ' ', sample_rate)
        
        
        float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max
        #print("new:", len(float_buffer))

        #print(buffer)
        #print(asr(float_buffer))
        t = await speech2Text(float_buffer)
        #print(t)
        await prompt(t['text'])
        # if not isopen:
        #     pya = pyaudio.PyAudio()
        #     stream = pya.open(format=pya.get_format_from_width(width=2), channels=num_channels, rate=sample_rate, output=True)


        #print("audio frame:", len(frame.to_ndarray()[0])) # multi channels
        #print(frame.to_ndarray().dtype)
        #print("audio frame:", frame.to_ndarray())

        #sa.play_buffer(buffer, num_channels, 2, sample_rate)

        # stream.write(buffer)

        #print(buffer)
        # clear buffer
        #print("--- %s seconds ---" % (time.time() - start_time))
    audio = []

#dirty
async def handle_audio(track):
    global capturingAudio
    global audio
    global sample_rate
    global num_channels
    
    # https://stackoverflow.com/questions/31674416/python-realtime-audio-streaming-with-pyaudio-or-something-else
    isopen = False

    while True:        
        try:
            frame = await track.recv()
            continue

            if not capturingAudio:
                continue # drop frame

            #continue

            sample_rate = frame.sample_rate
            #print('sample_rate:', sample_rate)
            num_channels = len(frame.layout.channels)
            #print('frame shape:', frame.to_ndarray().shape)
            audio.append(frame.to_ndarray())

            # if len(audio) < 250:
            #     continue

            # print("trying asr")


            #audio = []
        except MediaStreamError:
            # This exception is raised when the track ends
            break

    # stream.stop_stream()
    # stream.close()
    # logger.info(f"Exited audio processing loop")

from multiprocessing import Process, Queue
import asyncio

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
            #asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device=0)
            asr = pipeline("automatic-speech-recognition",model="openai/whisper-large-v3",device='cpu')

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

w = Worker()

async def speech2Text(float_buffer):
    return await w.send('stt',(float_buffer,))