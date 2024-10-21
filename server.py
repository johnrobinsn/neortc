# server.py
# Simple asyncio signal server for multi webrtc sessions with agents

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

from config import config
neortc_secret = config.get('neortc_secret')

from os import kill, getpid
from ssl import SSLContext, PROTOCOL_TLS_SERVER
from asyncio import create_task
from aiohttp import web
# from aiohttp.web_runner import GracefulExit
from socketio import AsyncServer

import asyncio

# from auth_neortc import neortc_secret
# from openai_agent.agent import start as start_oai

peers = {} 

sio = AsyncServer(cors_allowed_origins='*')

async def periodic():
    while True:
        peerkeys = list(peers.keys())
        print(len(peerkeys))
        peer = '' if len(peerkeys) < 1 else peerkeys[0]
        print('periodic',peers,peer)
        await sio.emit('blahblah', room=peer)
        await asyncio.sleep(1)

@sio.event
async def forwardMessage(sid,target_sid,m):
    await sio.emit('onMessage', (sid,m,), room=target_sid)

@sio.event
async def sendText(sid,target_sid,t):
    await sio.emit('sendText', (sid,t,), room=target_sid)

@sio.event
async def error(e):
    log.error('socket io error:', e)

@sio.event
async def captureAudio(sid,target_sid,f):
    print('server.py captureAudio')
    await sio.emit('captureAudio', (sid,f,), room=target_sid)

@sio.event
async def connect(sid,env,auth):
    print("in bound connection")
    token = auth.get('token','')
    if neortc_secret and token != neortc_secret:
        print("auth failed; disonnecting")
        return False
    else:
        await sio.emit('peersChanged', (peers,))

# forward to agent
@sio.event
async def getContexts(sid,target_sid):
    log.info("Forwarding getContexts")
    await sio.emit('getContexts',sid,room=target_sid)

# forward reply back to client
@sio.event
async def getContextsResult(sid,target_sid,contexts):
    log.info("forwarding reply getContextsResult %s", target_sid)
    await sio.emit('getContextsResult',(sid,contexts,),room=target_sid if target_sid is not '' else None)

@sio.event
async def broadcaster(sid,info):
    print('broadcaster sid:',sid,' info:', info)
    info['id'] = sid
    peers[sid] = info
    await sio.emit('peersChanged', (peers,))
    await sio.emit('broadcaster')

@sio.event
async def watcher(sid):
    print('watcher: ', peers)
    await sio.emit('peersChanged', (peers,))

@sio.event
async def watch(sid, target_sid):
    await sio.emit('watcher', (sid,), room=target_sid)

@sio.event
async def offer(sid, target_sid, message, context):
    await sio.emit('offer', (sid, message, context), room=target_sid)

@sio.event
async def answer(sid, target_sid, message):
    await sio.emit('answer', (sid, message), room=target_sid)

@sio.event
async def candidate(sid, target_sid, message):
    log.info("Forwarding candidate; target: %s, message: %s", target_sid, message)
    await sio.emit('candidate', (sid, message), room=target_sid)

@sio.event
async def disconnect(sid):
    log.info('socket disconnect: %s', sid)
    if sid in peers:
        del peers[sid]
        await sio.emit('peersChanged', (peers,))
    await sio.emit('disconnectPeer', (sid,))

routes = web.RouteTableDef()

@routes.get('/')
async def handle_get(request):
    token = request.query.get('token', '')
    if neortc_secret and token != neortc_secret:
        return web.Response(text=".")
    else:
        return web.FileResponse('./static/index.html')

routes.static('/', './static', show_index=False)

app = web.Application()
sio.attach(app)   
app.add_routes(routes)


async def start_background_tasks2(app):
    print("starting tasks")
    app['tasks'] = []
    app['tasks'].append(create_task(periodic()))

async def start_background_tasks(app):
    print("starting tasks")
    app['tasks'] = []
    app['tasks'].append(create_task(start_oai()))

async def cleanup_background_tasks(app):
    print("cleaning up tasks")
    for t in app['tasks']:
        log.info('shutting down openai')
        t.cancel()
        await t
    log.info('done shutting down tasks')

if False:
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
else:
    # app.on_startup.append(start_background_tasks2)
    pass



async def on_shutdown(app):
    # force shutdown on ctrl-c
    print('')
    kill(getpid(), 15)

app.on_shutdown.append(on_shutdown)

default_host='*'
default_port = config.get('neortc_port')

ssl_context = SSLContext(PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain('./mycert.pem','./mykey.pem')

print("Starting webserver port:", default_port)
web.run_app(app, host=default_host, port=default_port, ssl_context=ssl_context)
