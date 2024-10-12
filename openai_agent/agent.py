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
    from tts_openai import TTSTrack
if enableLLM:
    from llm_openai import prompt, setMessageListener

class Context:
    def __init__():
        pass

def getContext(id):
    pass

class Peer:
    peerIndex = 0
    def __init__(self):
        Peer.peerIndex += 1
        self.peerName = f'peer{Peer.peerIndex}'
        self.pc = RTCPeerConnection(configuration=RTCConfiguration([
                RTCIceServer("stun:stun.l.google.com:19302"),
                #RTCIceServer("turn:turnserver.cidaas.de:3478?transport=udp", "user", "pw"),
                ]))
        self.setupPeer()

    def setupPeer(self):
        
        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            print('*** channel created')
            @channel.on('message')
            async def on_message(message):
                print(f'Received Message: {message}')
                if isinstance(message,str) and message=='ping':
                    channel.send("pong")

        # add media tracks
        if enableTTS:
            self.ttsTrack = TTSTrack()
            # self.pc.addTrack(tts_track())
            self.pc.addTrack(self.ttsTrack.ttsTrack)

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log.info(f"*** peer.connectionState = {self.pc.connectionState}")
            if self.pc.connectionState == 'closed':
                await self.sio.disconnect() #close()

        # Log ICE gathering state
        @self.pc.on("icegatheringstatechange")
        async def on_ice_gathering_state_change():
            log.info(f"ICE gathering state changed: {self.pc.iceGatheringState}")

        @self.pc.on("icecandidate")
        async def on_icecandidate(candidate):
            log.info('pc on ice candidate %s %s', self.sio, candidate)
            self.sio.emit("candidate", (self.sio, candidate), room=broadcaster)

        # Note: aiortc leaks if we don't consume the media frames...
        async def drop_media_data(track):
                try:
                    while True:
                        await track.recv()
                except:
                    pass

        @self.pc.on("track")
        async def on_track(track):
            # guard to do this just once... 
            if (track.kind == 'audio'):
                # pass
                log.info('Audio track received')
                asyncio.create_task(handle_audio(track))
            elif (track.kind == 'video'):
                # asyncio.create_task(handle_video(track))
                asyncio.create_task(drop_media_data(track))

                print("******** before creating datachannel")
                channel = self.pc.createDataChannel('chat')

                @channel.on("open")
                async def on_open():
                    print("Data channel is open")
                    channel.send("ping")

                @channel.on("message")
                async def on_message(message):
                    print(f"Received message: {message}")                

            @track.on('ended')
            async def on_ended():
                log.info('track ended')
        #await sio.emit('watch', id)


class Agent:
    def __init__(self):
        # self.sio = AsyncClient(ssl_verify=False,logger=True,engineio_logger=True)
        self.sio = AsyncClient(ssl_verify=False)
        self.connected = False
        # self.peer = None
        self.callbacks()
        self.watch_sid = None
        self.contexts = {
            'one': {},
            'two': {},
            'three': {}
        }
        # self.peers = {}
        self.peer = Peer()

    async def start(self,signal_server=SIGNAL_SERVER):
        # print('signal server:', signal_server)
        log.warning('signal server: %s', signal_server)
        try:
            startWhisper()
            await self.sio.connect(signal_server, auth={'token':neortc_secret},transports=['websocket'])
            await self.sio.wait()
        # except KeyboardInterrupt:
        #     print('Exiting OpenAI Agent...')
        except Exception as e:
            print('Exception received', e)        

    def callbacks(self):
        @self.sio.event
        async def connect():
            log.info("ws onconnect")

            log.info('Watcher Connected')
            await self.sio.emit('watcher')

            displayName = 'oai_agent'

            # print('broadcasting:', displayName)
            await self.sio.emit("broadcaster", {'displayName':displayName});
            self.connected = True


        @self.sio.event
        async def getContexts(id):
            log.info("agent:getContexts")
            await self.sio.emit("getContextsResult", (id, list(self.contexts.keys()),))

        @self.sio.event
        async def sendText(t):
            m = re.search(r'\w*\\([^ ]+) (.*)',t)
            if m:
                c = m.group(1)
                if c == 'say':
                    await self.peer.ttsTrack.say(m.group(2))
                else:
                    log.warning('unknown escaped command:', c)
            else:
                await prompt(t)

        @self.sio.event
        async def captureAudio(f):
            await setCaptureAudio(f)
            # pass

        @self.sio.event
        async def broadcaster():
            await self.sio.emit('watcher')

        @self.sio.event
        async def offer(id,message):  # initiating message; id is watching
            # global watch_sid


            async def onMessage(m):
                await self.sio.emit('forwardMessage', (self.watch_sid,m,))
                if m['role'] == 'assistant' and m['content']:
                    await self.peer.ttsTrack.say(m['content'][0]['text'])

            log.info('offer received %s, %s', id, message)

            #self.setupPeer()
            self.watch_sid = id
            if enableLLM:
                setMessageListener(onMessage)

            # if self.peer:
            #     description = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
            #     await self.peer.setRemoteDescription(description)

            #     # add tracks if we have them

            #     await self.peer.setLocalDescription(await self.peer.createAnswer())

            #     local_description = self.peer.localDescription
            #     await self.sio.emit("answer", (id,{"sdp": local_description.sdp, "type": local_description.type},))
            if self.peer:
                description = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
                await self.peer.pc.setRemoteDescription(description)

                # add tracks if we have them

                await self.peer.pc.setLocalDescription(await self.peer.pc.createAnswer())

                local_description = self.peer.pc.localDescription
                await self.sio.emit("answer", (id,{"sdp": local_description.sdp, "type": local_description.type},))

        @self.sio.event
        async def candidate(id,message):
            log.info('candidate received, %s, %s', id, message)
            if self.peer.pc != None:
                c = candidate_from_sdp(message["candidate"].split(":", 1)[1])
                c.sdpMid = message['sdpMid']
                c.sdpMLineIndex = message['sdpMLineIndex']
                await self.peer.pc.addIceCandidate(c)

        @self.sio.event
        async def disconnectPeer(id):
            log.info('disconnectPeer')
            # if self.broadcaster == id and self.peer != None:
            #     #self.peer.term()
            #     self.peer = None
            #     self.broadcaster = None

        @self.sio.event
        async def disconnect():
            log.info('Watcher Disconnected')
            setMessageListener(None)
            # if self.peer is not None:
            #     #self.peer.term()
            #     self.peer = None



# match = '.'
# broadcaster = None
# peer = None

# connected=False





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
        agent = Agent()
        asyncio.run(agent.start(signal_server))
    except Exception as e:
        print(traceback.format_exc())
        print('Exception2 received', e)

