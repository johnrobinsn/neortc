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

# logging.getLogger("transformers").setLevel(logging.ERROR)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import threading
from threadworker import ThreadWorker
from multiprocessing import Queue
import asyncio

import numpy as np

from pydub import AudioSegment

from stt_whisper_sync import STT

#TODO... 
# state change for when model is loaded and ready

class AsyncSTT(STT):
    '''
    AsyncSTT is a subclass of STT that provides a non-blocking API for doing transcripiton
    using the whisper model.
    '''
    def __init__(self, sample_rate=48000, num_channels=2, captureDir='test/blah', enableFullCapture=False):
        super().__init__(sample_rate,num_channels,captureDir,enableFullCapture)
        self.w = ThreadWorker(supportOutQ=False)
        self.shared_lock = threading.Lock()
        self.loop = asyncio.get_event_loop()

    def __del__(self):
        self.w.stop()

    @staticmethod
    def preload_shared_resources():
        # preload the shared resources used by the parent class
        STT.preload_shared_resources()

    async def asyncNotifyListeners(self, *result):
        for listener in self.listeners:
            if asyncio.iscoroutinefunction(listener):
                await listener(self,*result)
            else:
                listener(self,*result)

    def notifyListeners(self, eventType, data):
        asyncio.run_coroutine_threadsafe(self.asyncNotifyListeners(eventType, data), self.loop)

    def super_processBuffer(self, buffer):
        # use a lock to protect the shared resources used by the parent class
        with self.shared_lock:
            return super().processBuffer(buffer)
        
    def processBuffer(self, buffer):
        # dispatch processBuffer to threadworker
        self.w.add_task(self.super_processBuffer, buffer)

if __name__ == "__main__":

    stt = AsyncSTT(sample_rate=48000,num_channels=2,captureDir='test2/blah')
    AsyncSTT.preload_shared_resources()

    stt.addListener(lambda s,e,d: print(f"STT Event: {e} {d}"))

    audio_file = '/home/jr/Downloads/fullcapture.mp3'
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
        # w.inQ.put(('process',(chunk,stt.id,'test/blah')))
        stt.processBuffer(chunk)

    # asyncio.get_event_loop().run_until_complete(stopWhisper())
    # asyncio.get_event_loop().run_forever()
    try:
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        pass
    # asyncio
    # await stopWhisper()
    print('Done')
