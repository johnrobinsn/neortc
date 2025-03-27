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

import weakref

import atexit

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
        # self._finalizer = weakref.finalize(self, self._cleanup)
        # # Register the atexit cleanup with a weak reference
        # self_ref = weakref.ref(self)
        # atexit.register(lambda: self_ref() and self_ref()._cleanup())

    def _cleanup(self):
        # cleanup the shared resources used by the parent class
        print('AsyncSTT cleanup')
        self.w.stop()

    # def __del__(self):
    #     self.w.stop()

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

    def super_flush(self, newCaptureDir=None):
        # use a lock to protect the shared resources used by the parent class
        with self.shared_lock:
            return super().flush(newCaptureDir)

    def flush(self,newCaptureDir=None):
        # dispatch flush to threadworker
        self.w.add_task(self.super_flush, newCaptureDir)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stt_whisper.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    # audio_file = '/home/jr/Downloads/fullcapture.mp3'

    loop = asyncio.get_event_loop()

    try:
        audio_segment = AudioSegment.from_mp3(audio_file)
    except Exception as e:
        print('Failed to load audio file:',e)
        sys.exit(1)

    stt = AsyncSTT(sample_rate=48000,num_channels=2,captureDir='test/blah',enableFullCapture=True)
    AsyncSTT.preload_shared_resources()

    # stt.addListener(lambda s,e,d: print(f"STT Event: {e} {d}"))
    def listener(s,e,d):
        print(f"STT Event: {e} {d}")
        if e == 'flushed':
            loop.stop()
    stt.addListener(listener)

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
    print('before flush')
    stt.flush()

    # asyncio.get_event_loop().run_until_complete(stopWhisper())
    # asyncio.get_event_loop().run_forever()
    # async def shutdown(loop):
    #     await asyncio.sleep(30)
    #     loop.stop()
    #     stt.flush()


    # loop.create_task(shutdown(loop))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass

    async def shutdown(loop):
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]

        [task.cancel() for task in tasks]

        print(f"Cancelling {len(tasks)} outstanding tasks")
        await asyncio.gather(*tasks, return_exceptions=True)
        loop.stop()

    loop.run_until_complete(shutdown(loop))
    loop.close()
    stt._cleanup()  # TODO... hack to cleanup the threadworker

    # asyncio
    # await stopWhisper()
    # try:
    #     loop.run_until_complete(main())
    # except KeyboardInterrupt:
    #     pass        
    # finally:
    #     loop.run_until_complete(shutdown(loop))
    #     loop.close()

    print('Done')
