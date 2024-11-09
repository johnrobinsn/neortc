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

# import uuid
# import re
import os
import asyncio
from socketio import AsyncClient
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

from aconfig import config

# TODO silent error if connection fails
SIGNAL_SERVER = f'wss://localhost:{config.get("neortc_port")}'
neortc_secret = config.get('neortc_secret')

enableTTS = True
enableLLM = True
enableSTT = True

# for each peer
if enableSTT:
    from stt_whisper import startWhisper, STT
if enableTTS:
    from tts_openai import TTSTrack

# for each context
if enableLLM:
    from llm_openai import LLM

import json
class Peer:
    peerIndex = 0
    def __init__(self, context, key):
        Peer.peerIndex += 1
        self.peerName = f'peer{Peer.peerIndex}'
        self.key = key
        print('Creating New Peer: ', self.peerName)
        self.pc = RTCPeerConnection(configuration=RTCConfiguration([
                RTCIceServer("stun:stun.l.google.com:19302"),
                #RTCIceServer("turn:turnserver.cidaas.de:3478?transport=udp", "user", "pw"),
                ]))
        self.setupPeer()
        self.context = None
        
        self.stt = STT() # could probably defer this until really needed
        self.dataChannel = None
 
        async def onMetaDataChanged(m):
            print('********** onMetaDataChanged', m)
            if self.dataChannel and self.dataChannel.readyState == 'open':
                print('sending')
                msg = {'t':'onMetaDataChanged','p':m}
                self.dataChannel.send(json.dumps(msg))
            else:
                print('data channel not open')                

        # context modified update connected peers
        async def onMessage(m):
            # print("**** Message: ", m)
            if self.dataChannel and self.dataChannel.readyState == 'open':
                msg = {'t':'appendLog','p':m}
                self.dataChannel.send(json.dumps(msg))
            else:
                log.error(f"datachannel closed {self.peerName}")
                return # bail out
            if self.ttsTrack and m['role'] == 'assistant' and m['content']:
                await self.ttsTrack.say(m['content'][0]['text'])  

        self.listener = onMessage
        self.metaDataListener = onMetaDataChanged              
        agent.peers[self.key] = self

    def __del__(self):
        log.info(f'Peer finalized: {self.key}')

    def setContext(self,context):
        print('setting context on peer', context)
        if self.context:
            if self.listener:
                self.context.delListener(self.listener)
            if self.metaDataListener:
                self.context.delMetaDataListener(self.metaDataListener)

        self.stt.setLLM(None)
        self.context = None
        if context:
            self.context = context
            self.stt.setLLM(context)

            print('*** warning self.listener:', self.listener)
            context.addListener(self.listener)

            # send log to peer
            self.replaceLog()

    def replaceLog(self):
        if self.dataChannel and self.dataChannel.readyState == 'open':
            msg = {'t':'onMetaDataChanged','p':self.context.getMetaData()}
            self.dataChannel.send(json.dumps(msg))            
            msg = {'t':'replaceLog','p':self.context.prompt_messages}
            self.dataChannel.send(json.dumps(msg))

    def setupPeer(self):
        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            self.dataChannel = channel
            print('*** channel created',channel.readyState)
            if self.context:
                self.context.addMetaDatalistener(self.metaDataListener)            
            self.replaceLog()
            # probably want to deter all listeners until channel is open
            @channel.on('open')
            async def on_open():
                print("dc is open: ", channel.readyState)

            @channel.on('message')
            async def on_message(message):
                print(f'Received Message: ', message)
                print(f'channel state: {channel.readyState}')

                o = json.loads(message)
                print('onMessage:', o)
                if o['t'] == 'sendText':
                    await self.context.prompt(o['p'])                    
                elif o['t'] == 'setContext':
                    self.setContext(await agent.getContext(o['p']))
                elif o['t'] == 'captureAudio':
                    await self.stt.setCaptureAudio(o['p'])
                elif o['t'] == 'clearAudio':
                    self.ttsTrack.clearAudio()
                elif o['t'] == 'enableAudio':
                    self.ttsTrack.enableAudio(o['p'])

            @channel.on('close')
            async def on_close():
                print(f'channel closed: {channel.readyState}')

        # add media tracks
        if enableTTS:
            self.ttsTrack = TTSTrack()
            self.pc.addTrack(self.ttsTrack.getTrack())

        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            log.info(f"*** peer.connectionState = {self.pc.connectionState}")
            state = self.pc.connectionState

            if state == 'closed' or state == 'failed':
                self.setContext(None)
                del agent.peers[self.key]
                print('removing peer from agent', self.key)
                print('Total Peers: ', len(agent.peers))

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
                asyncio.create_task(self.stt.handle_audio(track))
            elif (track.kind == 'video'):
                # asyncio.create_task(handle_video(track))
                asyncio.create_task(drop_media_data(track))                        

            @track.on('ended')
            async def on_ended():
                log.info('track ended')

class Agent:

    nextContextId=0

    def __init__(self):
        # self.sio = AsyncClient(ssl_verify=False,logger=True,engineio_logger=True)
        self.sio = AsyncClient(ssl_verify=False)
        self.connected = False

        self.callbacks()
        self.watch_sid = None

        self.listener = None

        self.contexts = self.loadAll()
        self.peers = {}
    
    def loadAll(self):
        c = {}
        dir = 'openai_chats'
        try:
            files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            for index, file in enumerate(files, start=1):
                print(f"{index}. {file}")
                l = LLM()
                l.load(f'{dir}/{file}')
                c[l.id] = l
                async def onContextMetaDataChanged(m):
                    await self.updateContexts()
                l.addMetaDatalistener(onContextMetaDataChanged)
        except Exception as e:
            print(f"Error reading directory: {e}")
        return c

    async def getContext(self, cid):
        print('Getting context: ', cid)
        if cid in self.contexts:
            return self.contexts[cid]
        else:
            # if not cid:
            #     # create a context Str
            #     cid = f'context{Agent.nextContextId}'
            #     cid = str(uuid.uuid4())
            #     Agent.nextContextId += 1            
            # l = LLM(cid)
            l = LLM()
            self.contexts[l.id] = l
            async def onContextMetaDataChanged(m):
                # garbage collecting
                for k in list(self.contexts.keys()):
                    if self.contexts[k].dead:
                        del self.contexts[k]                
                await self.updateContexts()
            l.addMetaDatalistener(onContextMetaDataChanged)
            await self.updateContexts() # broadcast
            return l

    # async def createPeer(self,sid,contextStr = ''):
    #     print('Getting Peer for Client: ', sid)
    #     if sid in self.peers:
    #         return self.peers[sid]
    #     else:
    #         c = await self.getContext(contextStr)
    #         p = Peer(c,sid)
    #         p.setContext(c)
    #         return p

    async def createPeer(self,sid,key,contextStr = ''):
        print('***** creating a new peer')
        # if sid in self.peers:
        #     p = self.peers[sid]
        #     del self.peers[sid]
        #     p.key = None
        c = await self.getContext(contextStr)
        p = Peer(c,key)
        p.setContext(c)
        return p
        
    def getPeer(self,key):
        if (key in self.peers):
            return self.peers[key]
        else:
            print("ERROR: Unknown peer!")

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

    async def updateContexts(self,id=''):
        metaData = [v.getMetaData() for i,(k,v) in enumerate(self.contexts.items())]
        # metaData = []
        log.info("Updating contexts: %s", metaData)
        log.info("contexts: %s", self.contexts)
        # await self.sio.emit("getContextsResult", (id, list(self.contexts.keys()),))      
        await self.sio.emit("getContextsResult", (id, metaData,))    

    def callbacks(self):
        @self.sio.event
        async def connect():
            log.info("ws onconnect")

            log.info('Watcher Connected')
            await self.sio.emit('watcher')

            displayName = config.get('agent_name','default')

            # print('broadcasting:', displayName)
            await self.sio.emit("broadcaster", {'displayName':displayName});
            self.connected = True

        @self.sio.event
        async def getContexts(id):
            log.info("agent:getContexts")
            await self.updateContexts(id=id)

        # @self.sio.event
        # async def captureAudio(sid,f):
        #     peer = self.getPeer(sid)
        #     await peer.stt.setCaptureAudio(f)

        @self.sio.event
        async def broadcaster():
            await self.sio.emit('watcher')

        @self.sio.event
        async def offer(id,message,contextStr,key):  # initiating message; id is watching
            # global watch_sid
            print("**** requested context: ", contextStr)
            peer = await self.createPeer(id,key,contextStr)
            print("id:", id, " peer: ", peer)

            log.info('offer received %s, %s', id, message)

            self.watch_sid = id
            if peer:
                #message = json.loads(message)
                print('message:',message)
                description = RTCSessionDescription(sdp=message["sdp"], type=message["type"])
                await peer.pc.setRemoteDescription(description)

                # add tracks if we have them

                try:
                    await peer.pc.setLocalDescription(await peer.pc.createAnswer())
                except Exception as e:
                    print(f"Task failed with error: {e}")

                local_description = peer.pc.localDescription
                await self.sio.emit("answer", (id,{"sdp": local_description.sdp, "type": local_description.type},))

        @self.sio.event
        async def candidate(id,message,key):
            peer = self.getPeer(key)
            log.info('candidate received, %s, %s', id, message)
            if peer.pc != None:
                c = candidate_from_sdp(message["candidate"].split(":", 1)[1])
                c.sdpMid = message['sdpMid']
                c.sdpMLineIndex = message['sdpMLineIndex']
                await peer.pc.addIceCandidate(c)

        @self.sio.event
        async def disconnectPeer(id):
            log.info('disconnectPeer')

        @self.sio.event
        async def disconnect():
            log.info('Watcher Disconnected')

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
    signal_server = f"{config.get('agent_signal_server','')}?agent={config.get('agent_name','default')}"
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

