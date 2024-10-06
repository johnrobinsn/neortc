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
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

import re
import asyncio
from socketio import AsyncClient
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from aconfig import config

# TODO silent error if connection fails
SIGNAL_SERVER = f'wss://localhost:{config.get("neortc_port")}'
# from auth_neortc import neortc_secret
#from config import config
neortc_secret = config.get('neortc_secret')


enableTTS = True
enableLLM = True
enableSTT = True

if enableSTT:
    from stt_whisper import handle_audio, setCaptureAudio, startWhisper
if enableTTS:
    from tts_openai import tts_track,say
if enableLLM:
    from llm_openai import prompt, setMessageListener, getMessages

sio = AsyncClient(ssl_verify=False,logger=True,engineio_logger=True)

match = '.'
broadcaster = None
peer = None

connected=False

@sio.event
async def connect():
    global sio
    global connected
    log.info("ws onconnect")

    log.info('Watcher Connected')
    await sio.emit('watcher')

    displayName = 'oai_agent'

    # print('broadcasting:', displayName)
    await sio.emit("broadcaster", {'displayName':displayName});
    connected = True

@sio.event
def ping(sid):
    logging.info(f'Received ping from client: {sid}')

@sio.event
def pong(sid):
    logging.info(f'Received pong from client: {sid}')

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

    peer = RTCPeerConnection(configuration=RTCConfiguration([
            RTCIceServer("stun:stun.l.google.com:19302"),
            #RTCIceServer("turn:turnserver.cidaas.de:3478?transport=udp", "user", "pw"),
            ]))

    # add media tracks
    if enableTTS:
        peer.addTrack(tts_track())

    @peer.on("connectionstatechange")
    async def on_connectionstatechange():
        log.info(f"*** peer.connectionState = {peer.connectionState}")
        if peer.connectionState == 'closed':
            await sio.disconnect() #close()

    # Log ICE gathering state
    @peer.on("icegatheringstatechange")
    async def on_ice_gathering_state_change():
        log.info(f"ICE gathering state changed: {peer.iceGatheringState}")

    @peer.on("icecandidate")
    async def on_icecandidate(candidate):
        log.info('pc on ice candidate %s %s', sio, candidate)
        sio.emit("candidate", (sio, candidate), room=broadcaster)

    @peer.on("track")
    async def on_track(track):
        # guard to do this just once... 
        if (track.kind == 'audio'):
            # pass
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
    # pass

@sio.event
async def broadcaster():
    await sio.emit('watcher')

watch_sid = None

@sio.event
async def offer(id,message):  # initiating message; id is watching
    global watch_sid
    # log.info('offer received', id, message)

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
    log.info('candidate received, %s, %s', id, message)
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
    log.info('Watcher Disconnected')
    setMessageListener(None)
    # if self.peer is not None:
    #     #self.peer.term()
    #     self.peer = None

# @sio.event
# async def blahblah():
#     print('blahblah')

async def start(signal_server=SIGNAL_SERVER):
    # print('signal server:', signal_server)
    log.warn('signal server: %s', signal_server)
    try:
        startWhisper()
        await sio.connect(signal_server, auth={'token':neortc_secret},transports=['websocket'])
        await sio.wait()
    # except KeyboardInterrupt:
    #     print('Exiting OpenAI Agent...')
    except Exception as e:
        print('Exception received', e)


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

import traceback
if __name__ == "__main__":
    signal_server = config.get('agent_signal_server','')
    if not signal_server:
        log.error('No signal server defined')
        exit(0)
    else:
        log.info('Attempting connection to %s', signal_server)
    try:
        asyncio.run(start(signal_server))
    except Exception as e:
        print(traceback.format_exc())
        print('Exception2 received', e)

