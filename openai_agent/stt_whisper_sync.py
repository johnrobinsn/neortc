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

import numpy as np
import torch
from transformers import pipeline

from samplerate import resample
from pydub import AudioSegment
from typing import Literal
from enum import Enum, auto
from typing import Dict, Tuple
from dataclasses import dataclass
from typing import Callable, Optional

# Guidance
# This is intended to be the base level STT module
# No async, no threads, blocking api

class STT:
    def __init__(self, sample_rate, num_channels,captureDir=None,enableFullCapture=False):
        self.whisper_sample_rate = 16000
        self.silero_sample_rate = self.whisper_sample_rate # code assumes this to be same as whisper

        self.listeners = []
        
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self.captureDir = captureDir 

        self.statemachine = self.createStateMachine()
        self.segment = []
        self.enableFullCapture = enableFullCapture
        self.fullCapture = []

    shared_resources = {}

    @staticmethod
    def preload_shared_resources():
        if 'vad' not in STT.shared_resources:
            log.info('Loading Silero VAD')
            model, _utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad',
                                        force_reload=True)
            STT.shared_resources['vad'] = model
            log.info('Loaded Silero VAD')

        if 'asr' not in STT.shared_resources:
            log.info('Loading Whisper')
            asr = pipeline("automatic-speech-recognition",model="openai/whisper-tiny.en",device='cpu')
            STT.shared_resources['asr'] = asr
            log.info('Loaded Whisper')

    @staticmethod
    def unload_shared_resources():
        STT.shared_resources = {}    
    
    def addListener(self,listener):
        self.listeners.append(listener)

    def removeListener(self,listener):
        self.listeners.remove(listener)

    def notifyListeners(self,eventType,data):
        for l in self.listeners:
            l(self,eventType,data)

    class Event(Enum):
        VOICE_WAS_DETECTED = auto()
        VOICE_NOT_DETECTED = auto()
    class State(Enum):
        IDLE = auto()       # Waiting for voice
        SEGMENT_0 = auto()  # First voice segment
        SEGMENT_N = auto()  # Subsequent voice segments
        SILENCE_0 = auto()  # First silence segment
        SILENCE_1 = auto()  # Second silence segment
        SILENCE_2 = auto()  # Third silence segment                

    def createStateMachine(self):
        stt = self

        @dataclass
        class Transition:
            next_state: STT.State
            action: Optional[Callable] = None

        class StateMachine:
            # nonlocal process_segment
            def __init__(self):

                def process_segment(s,b):
                    buffer = np.concatenate(s, axis=0)
                    float_buffer = buffer.astype(np.float32) / np.iinfo(np.int16).max

                    if stt.captureDir is not None:
                        try:
                            os.makedirs(stt.captureDir, exist_ok=True)
                            now = datetime.datetime.now()

                            filename = f'{stt.captureDir}/{now.strftime("%Y%m%d-%H%M%S")}.mp3'
                            audio_segment = AudioSegment(
                                buffer.tobytes(), 
                                frame_rate=16000,
                                sample_width=2,  # 2 bytes for 16-bit audio
                                channels=1
                            )

                            # Export the AudioSegment to an MP3 file
                            audio_segment.export(filename, format="mp3")
                        except Exception as e:
                            log.error(f"Error saving audio file (segment): {e}")

                    asr = STT.shared_resources['asr']
                    t = asr(float_buffer)

                    stt.notifyListeners('voice_not_detected',None)     
                    stt.notifyListeners('final_transcript',t['text'])  #todo stt id not being passed
                    s.clear()

                def voice_was_detected(s,b):
                    s.append(b)
                    stt.notifyListeners('voice_was_detected',None)               

                self.state = STT.State.IDLE
                
                # Define transitions with actions
                self.transitions: Dict[Tuple[STT.State, STT.Event], Transition] = {
                    (STT.State.IDLE, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.IDLE, 
                        lambda s, b: s.clear()
                    ),            
                    (STT.State.IDLE, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_0, 
                        lambda s, b: s.append(b)
                    ),
                    (STT.State.SEGMENT_0, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.IDLE,
                        lambda s, b: s.clear()
                    ),
                    (STT.State.SEGMENT_0, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_N,
                        lambda s, b: voice_was_detected(s,b)
                    ),            
                    (STT.State.SEGMENT_N, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.SILENCE_0,
                        lambda s, b: s.append(b)
                    ),
                    (STT.State.SEGMENT_N, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_N,
                        lambda s, b: s.append(b)
                    ),            
                    (STT.State.SILENCE_0, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.SILENCE_1,
                        lambda s, b: s.append(b)
                    ),
                    (STT.State.SILENCE_0, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_N,
                        lambda s, b: s.append(b)
                    ),            
                    (STT.State.SILENCE_1, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.SILENCE_2,
                        lambda s, b: s.append(b)
                    ),
                    (STT.State.SILENCE_1, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_N,
                        lambda s, b: s.append(b)
                    ),
                    (STT.State.SILENCE_2, STT.Event.VOICE_NOT_DETECTED): Transition(
                        STT.State.IDLE,
                        lambda s, b: process_segment(s,b)
                    ),
                    (STT.State.SILENCE_2, STT.Event.VOICE_WAS_DETECTED): Transition(
                        STT.State.SEGMENT_N,
                        lambda s, b: s.append(b)
                    )
                }
            
            def handle_event(self, event: STT.Event, *args):
                if (self.state, event) not in self.transitions:
                    log.error(f"Invalid transition: Cannot handle event {event.name} in state {self.state.name}")
                    return
                
                transition = self.transitions[(self.state, event)]
                old_state = self.state
                self.state = transition.next_state
                
                # log.info(f"Transition: {old_state.name} -> {self.state.name} [Event: {event.name}]")

                # Execute the transition action if it exists
                if transition.action:
                    transition.action(*args)

        return StateMachine()
    
    def flush(self,newCaptureDir=None):
        if self.captureDir and self.enableFullCapture and len(self.fullCapture) > 0:
            log.info('stt: saving fullcapture')
            try:
                buffer = np.concatenate(self.fullCapture, axis=1)

                os.makedirs(self.captureDir, exist_ok=True)
                filename = f'{self.captureDir}/fullcapture.mp3'
                audio_segment = AudioSegment(
                    buffer.tobytes(), 
                    frame_rate=self.sample_rate,#16000,
                    sample_width=2,  # 2 bytes for 16-bit audio
                    channels=self.num_channels#1
                )

                # Export the AudioSegment to an MP3 file
                audio_segment.export(filename, format="mp3")
            except Exception as e:
                log.error(f"Error saving audio file (fullcapture): {e}")         

        self.segment = []
        self.fullCapture = []
        self.statemachine.state = STT.State.IDLE

        if newCaptureDir:
            self.captureDir = newCaptureDir

    def processBuffer(self,buffer):
        STT.preload_shared_resources()
        if self.enableFullCapture:
            self.fullCapture.append(buffer)

        buffer = buffer.mean(axis=0) #warning implicit float conversion
        if self.num_channels == 2:
            buffer = ((buffer[::2] + buffer[1::2])/2).astype(np.int16)  # convert to mono by averaging channels
        ratio = self.whisper_sample_rate / self.sample_rate
        buffer = resample(buffer, ratio, 'sinc_best')
        buffer = buffer.astype(np.int16)

        float_buffer2 = buffer.astype(np.float32) / np.iinfo(np.int16).max
        float_buffer2 = float_buffer2.reshape(-1,512)

        # silero vad
        vad = STT.shared_resources['vad']
        p = vad(torch.from_numpy(float_buffer2),self.silero_sample_rate)
        p = torch.all(p<0.25)

        # Silence detection
        rms = np.sqrt(np.mean(buffer.astype(np.float32)**2))

        silence_threshold = 5  # Adjust this threshold as needed
        
        is_silent = rms < silence_threshold

        if not is_silent:
            is_silent = p # noisy but no voice detected

        self.statemachine.handle_event(STT.Event.VOICE_NOT_DETECTED if is_silent else STT.Event.VOICE_WAS_DETECTED,self.segment,buffer)

if __name__ == "__main__":

    stt = STT(sample_rate=48000, num_channels=2)

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

        stt.processBuffer(chunk)

    print('Done')
