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

import os
import signal
import traceback
import json
import asyncio
from asyncio import sleep
from time import time
from socketio import AsyncClient
from aiortc import RTCSessionDescription, RTCPeerConnection, RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av.packet import Packet
from fractions import Fraction
import numpy as np
from termcolor import colored

import datetime

from aconfig import config

# TODO silent error if connection fails
SIGNAL_SERVER = f'wss://localhost:{config.get("neortc_port")}'
neortc_secret = config.get('neortc_secret')

enableTTS = True
enableLLM = True
enableSTT = True

# for each peer
if enableSTT:
    from stt_whisper import AsyncSTT
if enableTTS:
    # from tts_openai import TTS_OpenAI
    from tts_kokoro import Async_TTS_Kokoro

# for each context
if enableLLM:
    from llm import LLM

class TTSTrack:
    def __init__(self):
        def opus_frame_handler(opus_frame):
            if not self.muted:
                self.packetq.insert(0,opus_frame)  # TODO should make contract be opus_frame, channels, ?? sample_rate
        # self.tts = TTS_OpenAI(opus_frame_handler=opus_frame_handler)
        self.tts = Async_TTS_Kokoro(opus_frame_handler=opus_frame_handler)
        self.text = ''
        self.muted = False
        self.packetq = []
        self.audioEnabled = False
        self.next_pts = 0
        self.silence_duration = 0.02

        self.time_base = 48000
        self.time_base_fraction = Fraction(1, self.time_base)        
        self._createTTSTrack()        

    def mute(self, f):
        self.muted = f

    def clearAudio(self):
        self.packetq.clear()
        self.text = ''
    
    def enableAudio(self,f):
        log.info('enableAudio: %s',f)
        if not f:
            self.clearAudio()
        self.audioEnabled = f
                
    async def open(self):
        self.text = ''

    async def write(self,text):
        self.text = self.text + text
        last_newline = max(self.text.rfind('\n'), self.text.rfind('. '))
        if last_newline != -1:
            await self._say(self.text[:last_newline + 1])
            self.text = self.text[last_newline + 1:]
        
    async def close(self):
        if self.text.strip():
            await self._say(self.text)
        self.text = ''        

    async def _say(self,text):        
        if self.audioEnabled:
            await self.tts.say(text)

    def getTrack(self):
        return self.ttsTrack
    
    def _createTTSTrack(self):
        def get_silence_packet(duration_seconds):
            chunk = bytes.fromhex('f8 ff fe')

            pkt = Packet(chunk)
            pkt.pts = self.next_pts
            pkt.dts = self.next_pts
            pkt.time_base = self.time_base_fraction

            pts_count = round(duration_seconds * self.time_base)
            self.next_pts += pts_count

            return pkt

        # if we we have audio queued deliver that; otherwise silence
        def get_audio_packet():
            if len(self.packetq) > 0:
                try:
                    chunk = self.packetq.pop()
                    duration = 20/1000 # assume 20ms
                    pts_count = round(duration * 48000) # 48000 is the sample rate and channel = 1

                    pkt = Packet(chunk)
                    pkt.pts = self.next_pts
                    pkt.dts = self.next_pts
                    pkt.time_base = self.time_base_fraction

                    self.next_pts += pts_count

                    return pkt,duration
                except:
                    pass # Ignore Empty exception

            return get_silence_packet(self.silence_duration), self.silence_duration    

        class tts_track(MediaStreamTrack):
            kind = "audio"
            
            def __init__(self):
                super().__init__()
                self.stream_time = None
                log.info('create tts_track')

            async def close(self):
                super().stop()

            async def recv(self):
                try: # exceptions that happen here are eaten... so log them
                    packet, duration = get_audio_packet()

                    if self.stream_time is None:
                        self.stream_time = time()

                    wait = self.stream_time - time()
                    await sleep(wait)

                    self.stream_time += duration
                    return packet
                except Exception as e:
                    log.error('Exception:', e)
                    raise

        self.ttsTrack = tts_track()    

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
        
        self.stt = AsyncSTT(sample_rate=48000,num_channels=2,enableFullCapture=True) # could probably defer this until really needed

        async def sttListener(s,e,d):
            if self.context:
                if e == 'final_transcript':
                    await self.context.prompt(d['transcript'])
                elif e == 'voice_was_detected':
                    await self.context.bargeIn()
                    # TODO do I need mute and clearAudio both?
                    self.ttsTrack.mute(True)
                    self.ttsTrack.clearAudio()
        self.stt.addListener(sttListener)

        self.dataChannel = None

        self.capturingAudio = False
        self.audio = []
 
        async def onMetaDataChanged(m):
            print('********** onMetaDataChanged', m)
            if self.dataChannel and self.dataChannel.readyState == 'open':
                print('sending')
                # TODO do I need both messages
                msg = {'t':'onMetaDataChanged','p':m}
                self.dataChannel.send(json.dumps(msg))
                await self.updateContexts()
            else:
                # TODO getting some occasional errors here
                print('data channel not open')                

        # context modified update connected peers
        async def onMessage(m):
            if m:
                if self.dataChannel and self.dataChannel.readyState == 'open':
                    # msg = {'t':'appendLog','p':m}
                    self.dataChannel.send(json.dumps(m))
                else:
                    log.error(f"datachannel closed {self.peerName}")
                    return # bail out
                # if self.ttsTrack and m['t'] == 'closeEntry' and m['role'] == 'assistant' and m['data']:
                #     await self.ttsTrack.say(m['data'])
                #
                if m['role'] == 'assistant':
                    if m['t'] == 'openEntry':
                        self.ttsTrack.mute(False)
                        await self.ttsTrack.open()
                    elif m['t'] == 'writeEntry':
                        await self.ttsTrack.write(m['data'])
                    elif m['t'] == 'closeEntry':
                        await self.ttsTrack.close()
            else:
                # TODO get rid of this code path
                pass

        self.listener = onMessage
        self.metaDataListener = onMetaDataChanged              
        agent.peers[self.key] = self

    def __del__(self):
        log.info(f'Peer finalized: {self.key}')

    def setContext(self,context):
        print('setting context on peer', context)
        if self.ttsTrack:
            self.ttsTrack.clearAudio()
        if self.context:
            if self.listener:
                self.context.delListener(self.listener)
            if self.metaDataListener:
                self.context.delMetaDataListener(self.metaDataListener)

        self.context = None
        if context:
            self.context = context

            print('*** warning self.listener:', self.listener)
            context.addListener(self.listener)
            context.addMetaDatalistener(self.metaDataListener)

            # send log to peer
            self.replaceLog()

    def replaceLog(self):
        if self.dataChannel and self.dataChannel.readyState == 'open':
            msg = {'t':'onMetaDataChanged','p':self.context.getMetaData()}
            self.dataChannel.send(json.dumps(msg))
            def filter(e):
                return e['role'] != 'system'
            filteredLog = [e for e in self.context.prompt_messages if filter(e)]          
            msg = {'t':'replaceLog','p':filteredLog}
            self.dataChannel.send(json.dumps(msg))

    def setupPeer(self):
        @self.pc.on("datachannel")
        async def on_datachannel(channel):
            self.dataChannel = channel
            print('*** channel created',channel.readyState)
            if self.context:
                self.context.addMetaDatalistener(self.metaDataListener)            
            self.replaceLog()
            # TODO probably want to deter all listeners until channel is open
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
                    await self.setCaptureAudio(o['p'])
                elif o['t'] == 'clearAudio':
                    self.ttsTrack.clearAudio()
                elif o['t'] == 'enableAudio':
                    self.ttsTrack.enableAudio(o['p'])
                elif o['t'] == 'getContexts':
                    await self.updateContexts()

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
            # TODO guard to do this just once... 
            if (track.kind == 'audio'):
                # pass
                log.info('Audio track received')
                asyncio.create_task(self.handle_audio(track))
            elif (track.kind == 'video'):
                # asyncio.create_task(handle_video(track))
                asyncio.create_task(drop_media_data(track))                        

            @track.on('ended')
            async def on_ended():
                log.info('track ended')

    async def setCaptureAudio(self,f):
        if self.capturingAudio == f:
            return
        print('Capturing Audio:', f)

        self.ttsTrack.clearAudio()
        if f:
            now = datetime.datetime.now()
            newCaptureDir = f'recordings/{now.strftime("%Y%m%d-%H%M%S")}'
            self.stt.captureDir = newCaptureDir # TODO
        else:
            self.stt.flush()

        self.capturingAudio = f

    async def handle_audio(self,track):
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
                # self.fullCapture.append(f)
                self.audio.append(f)

                if len(self.audio) >= 8: # 8=>320ms, 16=>640ms
                    buffer = np.concatenate(self.audio, axis=1)
                    # print('buffer:', buffer.shape, buffer.dtype, ' ', self.sample_rate)
                    # processed_buffer = await self.processor.process_audio(buffer, self.sample_rate)
                    self.stt.processBuffer(buffer)
                    self.audio = []  # Clear the buffer after processing

            except MediaStreamError:
                # This exception is raised when the track ends
                break

    async def updateContexts(self,id=''):
        metaData = [v.getMetaData() for i,(k,v) in enumerate(agent.contexts.items())]
        log.info("Updating contexts: %s", metaData)
        log.info("contexts: %s", agent.contexts)    
        try:
            # await self.sio.emit("getContextsResult", (id, metaData,))
            #             print('********** onMetaDataChanged', m)
            if self.dataChannel and self.dataChannel.readyState == 'open':
                msg = {'t':'onGetContextsResult','p':metaData}
                print('sending: ', msg)
                self.dataChannel.send(json.dumps(msg))
            else:
                print('data channel not open. could not send contexts')         
            
        except Exception as e:
            log.info('Exception: %s', e)

class Agent:

    nextContextId=0

    def __init__(self,promptFunc,agentName,promptStreamingFunc=None,initialPrompt=None):
        self.promptFunc = promptFunc
        self.promptStreamingFunc = promptStreamingFunc
        self.agentName = agentName
        # self.sio = AsyncClient(ssl_verify=False,logger=True,engineio_logger=True)
        self.sio = AsyncClient(ssl_verify=False)
        self.connected = False

        self.callbacks()
        self.watch_sid = None

        self.listener = None

        self.peers = {}
        self.initialPrompt = initialPrompt
        self.contexts = self.loadAll()

    def name(self):
        # TODO use hostname
        n = config.get('agent_name','default')
        n = f'{self.agentName}-{n}'      
        return n  
    
    def loadAll(self):
        c = {}
        dir = f'{self.agentName}_chats'
        try:
            files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
            for index, file in enumerate(files, start=1):
                print(f"{index}. {file}")
                l = LLM(self.promptFunc,self.agentName,self.promptStreamingFunc,self.initialPrompt)
                l.load(f'{dir}/{file}')
                c[l.id] = l
                # TODO can we get rid of all this in agent
                async def onContextMetaDataChanged(m):
                    # await self.updateContexts()
                    pass
                l.addMetaDatalistener(onContextMetaDataChanged)
        except Exception as e:
            print(f"Error reading directory: {e}")
        return c

    async def getContext(self, cid):
        print('Getting context: ', cid)
        if cid in self.contexts:
            return self.contexts[cid]
        else:
            l = LLM(self.promptFunc,self.agentName,self.promptStreamingFunc,self.initialPrompt)
            self.contexts[l.id] = l
            async def onContextMetaDataChanged(m):
                # garbage collecting
                for k in list(self.contexts.keys()):
                    if self.contexts[k].dead:
                        del self.contexts[k]                
                # await self.updateContexts()
            l.addMetaDatalistener(onContextMetaDataChanged)
            # await self.updateContexts() # broadcast
            return l

    async def createPeer(self,sid,key,contextStr = ''):
        print('***** creating a new peer')
        c = await self.getContext(contextStr)
        p = Peer(c,key)
        p.setContext(c)
        await p.updateContexts()
        return p
        
    def getPeer(self,key):
        if (key in self.peers):
            return self.peers[key]
        else:
            print("ERROR: Unknown peer!")

    async def start(self,signal_server=SIGNAL_SERVER):
        log.warning('signal server: %s', signal_server)
        try:
            AsyncSTT.preload_shared_resources()
            await self.sio.connect(signal_server, auth={'token':neortc_secret},transports=['websocket'])
            await self.sio.wait()
        except asyncio.CancelledError:
            log.info("Application is shutting down...")
        except Exception as e:
            log.info('Exception: %s', e) 
            print(traceback.format_exc())
        finally:
            await self.sio.disconnect()

            peerlist = list(self.peers.values())
            log.info('peerlist: %s', len(peerlist))

            for peer in peerlist:
                log.info(f"Closing peer connection")
                try:
                    await peer.pc.close()
                except Exception as e:
                    log.info('Exception: %s', e)

        log.info('Exiting Agent...')     

    def callbacks(self):
        @self.sio.event
        async def connect():
            log.info("ws onconnect")

            log.info('Watcher Connected')
            await self.sio.emit('watcher')

            await self.sio.emit("broadcaster", {'displayName':self.name()})
            self.connected = True

        @self.sio.event
        async def broadcaster():
            await self.sio.emit('watcher')

        @self.sio.event
        async def offer(id,message,contextStr,key):  # initiating message; id is watching
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
                    log.info('Exception: %s',e)

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

def handle_sigint(loop):
    print('Caught signal: SIGINT')
    # for task in asyncio.all_tasks(loop):
    #     task.cancel()

def startAgent(promptFunc,agentName,promptStreaming=None,initialPrompt=None):
    global agent # TODO ick

    loop = asyncio.get_event_loop()

    # Register the signal handler
    loop.add_signal_handler(signal.SIGINT, handle_sigint, loop)

    signal_server = f"{config.get('agent_signal_server','')}?agent={config.get('agent_name','default')}"
    if not signal_server:
        log.error('No signal server defined')
        exit(0)
    else:
        log.info('Attempting connection to %s', signal_server)
    try:
        agent = Agent(promptFunc,agentName,promptStreaming,initialPrompt)
        loop.run_until_complete(agent.start(signal_server))
    except asyncio.CancelledError:
        print("Application is shutting down2...")
    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt received, shutting down...")        
    except Exception as e:
        print('Exception2 received', e)
        print(traceback.format_exc())
    finally:
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        log.info("Agent exited cleanly")



