# agent.py
# A webrtc agent and allows for routing to different handlers for audio in, audio out, video in, video out
# can run inproc with the signal server and also can run standalone on a different machine than the signal server

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

import re
import asyncio
from socketio import AsyncClient
from aiortc import RTCSessionDescription, RTCPeerConnection
from aiortc.sdp import candidate_from_sdp

SIGNAL_SERVER = 'wss://localhost:8443?token=whiter@bbit'

enableTTS = True
enableLLM = True

from openai_agent.stt_whisper import handle_audio, setCaptureAudio
from openai_agent.tts_openai import tts_track,say
from openai_agent.llm_openai import prompt, setMessageListener, getMessages

sio = AsyncClient(ssl_verify=False)

match = '.'
broadcaster = None
peer = None

connected=False

@sio.event
async def connect():
    global sio
    global connected
    log.info('Watcher Connected')
    await sio.emit('watcher')

    displayName = 'oai_agent'

    await sio.emit("broadcaster", {'displayName':displayName});
    connected = True

# Note: aiortc leaks if we don't consume the media frames...
async def drop_media_data(track):
        try:
            while True:
                await track.recv()
        except:
            pass

def setupPeer():
    global peer
    global broadcaster
    global match
    global sio
    global connected

    peer = RTCPeerConnection()

    # add media tracks
    if enableTTS:
        peer.addTrack(tts_track())

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        log.info(f"*** peer.connectionState = {peer.connectionState}")
        if peer.connectionState == 'closed':
            await close()

    @peer.on("icecandidate")
    async def on_icecandidate(candidate):
        log.info('pc on ice candidate', sio, candidate)
        sio.emit("candidate", (sio, candidate), room=broadcaster)

    @peer.on("track")
    async def on_track(track):
        # guard to do this just once... 
        if (track.kind == 'audio'):
            asyncio.create_task(handle_audio(track))
        elif (track.kind == 'video'):
            # asyncio.create_task(handle_video(track))
            asyncio.create_task(drop_media_data(track))

        @track.on('ended')
        async def on_ended():
            log.info('track ended')
    #await sio.emit('watch', id)

async def onMessage(m):
    await sio.emit('forwardMessage', (watch_sid,m,))
    if m['role'] == 'assistant' and m['content']:
        await say(m['content'][0]['text'])


@sio.event
async def sendText(t):
    m = re.search(r'\w*\\([^ ]+) (.*)',t)
    if m:
        c = m.group(1)
        if c == 'say':
            await say(m.group(2))
        else:
            log.warning('unknown escaped command:', c)
    else:
        await prompt(t)

@sio.event
async def captureAudio(f):
    await setCaptureAudio(f)

@sio.event
async def broadcaster():
    await sio.emit('watcher')

watch_sid = None

@sio.event
async def offer(id,message):  # initiating message; id is watching
    global watch_sid
    log.info('offer received', id, message)

    setupPeer()
    watch_sid = id
    if enableLLM:
        setMessageListener(onMessage)

    if peer:
        description = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
        await peer.setRemoteDescription(description)

        # add tracks if we have them

        await peer.setLocalDescription(await peer.createAnswer())

        local_description = peer.localDescription
        await sio.emit("answer", (id,{"sdp": local_description.sdp, "type": local_description.type},))

@sio.event
async def candidate(id,message):
    log.info('candidate received', id, message)
    if peer != None:
        c = candidate_from_sdp(message["candidate"].split(":", 1)[1])
        c.sdpMid = message['sdpMid']
        c.sdpMLineIndex = message['sdpMLineIndex']
        await peer.addIceCandidate(c)

@sio.event
async def disconnectPeer(id):
    log.info('disconnectPeer')
    # if self.broadcaster == id and self.peer != None:
    #     #self.peer.term()
    #     self.peer = None
    #     self.broadcaster = None

@sio.event
async def disconnect():
    #log.info('Watcher Disconnected')
    setMessageListener(None)
    # if self.peer is not None:
    #     #self.peer.term()
    #     self.peer = None

async def start(*args):
    try:
        await sio.connect(SIGNAL_SERVER, transports=['websocket'])
        await sio.wait()
    except KeyboardInterrupt:
        print('Exiting OpenAI Agent...')

# async def start():
#     await sio.connect(SIGNAL_SERVER, transports=['websocket'])
#     #await sio.emit('sub.symbol', {'symbol': 'VDS_USDT'}, callback=callbk)
#     # sio.start_background_task(background_task, sio)
#     return sio.wait()

#await start()


# if __name__ == "__main__":
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(start())
# else:
#     loop = asyncio.get_event_loop()
#     loop.create_task(start())


# if __name__ == "__main__":
#     asyncio.run(start())