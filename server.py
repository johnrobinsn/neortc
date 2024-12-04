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
neortc_accounts = config.get('neortc_accounts',[])
neortc_accounts = {item['email']: item for item in neortc_accounts}

print('neortc_accounts:',neortc_accounts)
# Create a dictionary from an array of objects with one of the fields as a key
# def create_dict_from_array(array, key_field):
#     return 

# Example usage
# example_array = [
#     {'id': 1, 'name': 'Alice'},
#     {'id': 2, 'name': 'Bob'},
#     {'id': 3, 'name': 'Charlie'}
# ]

# example_dict = create_dict_from_array(example_array, 'id')
# print(example_dict)
print('neortc_secret:',neortc_secret)

from os import kill, getpid
from ssl import SSLContext, PROTOCOL_TLS_SERVER
from asyncio import create_task
from aiohttp import web

from socketio import AsyncServer

import asyncio

peers = {} 

sio = AsyncServer(cors_allowed_origins='*')

# async def periodic():
#     while True:
#         peerkeys = list(peers.keys())
#         print(len(peerkeys))
#         peer = '' if len(peerkeys) < 1 else peerkeys[0]
#         print('periodic',peers,peer)
#         await sio.emit('blahblah', room=peer)
#         await asyncio.sleep(1)

@sio.event
async def error(e):
    log.error('socket io error:', e)

import urllib
import secrets

@sio.event
async def connect(sid,env,auth):
    query_string = env.get('QUERY_STRING', '')
    query_params = urllib.parse.parse_qs(query_string)
    agentname = query_params.get('agent', [None])[0]

    print("in bound connection, ", agentname)
    token = auth.get('token','')
    print("socket token:",token,neortc_secret)
    if neortc_secret and token != neortc_secret:
        print("auth failed; disonnecting")
        return False
    else:
        await sio.emit('peersChanged', (peers,))

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
async def offer(sid, target_sid, message, context, key):
    await sio.emit('offer', (sid, message, context, key), room=target_sid)

@sio.event
async def answer(sid, target_sid, message):
    await sio.emit('answer', (sid, message), room=target_sid)

@sio.event
async def candidate(sid, target_sid, message, key):
    log.info("Forwarding candidate; target: %s, message: %s", target_sid, message)
    await sio.emit('candidate', (sid, message, key), room=target_sid)

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
    token = request.cookies.get('session_token')
    print('token:',token)
    if token != SESSION_TOKEN:
        # return web.Response(text="Unauthorized", status=401)
        return web.FileResponse('./static/index.html')
    else:
        return web.HTTPFound('/neortc')

#TODO needs to be fixed... hardcoded probably need a db for long lived sessions
SESSION_TOKEN = neortc_secret #secrets.token_urlsafe(32) # TODO needs to be fixed... hardcoded

@routes.post('/auth')
async def handle_auth(request):
    print('auth')
    data = await request.post()
    email = data.get('email')
    password = data.get('password')

    VALID_EMAIL = neortc_accounts.get(email, {}).get('email','')
    VALID_PASSWORD = neortc_accounts.get(email, {}).get('password','')
    
    if email == VALID_EMAIL and password == VALID_PASSWORD:
        response = web.HTTPFound('/neortc')
        response.set_cookie('session_token', SESSION_TOKEN, max_age=3600*24*365)
        return response
    else:
        return web.Response(text="Invalid credentials", status=401)

@routes.get('/neortc')
async def handle_get(request):
    token = request.cookies.get('session_token')
    if token != SESSION_TOKEN:
        return web.Response(text="Unauthorized", status=401)
    else:
        return web.FileResponse('./static/neortc.html')

routes.static('/', './static', show_index=False)

app = web.Application()
sio.attach(app)   
app.add_routes(routes)


# async def start_background_tasks2(app):
#     print("starting tasks")
#     app['tasks'] = []
#     app['tasks'].append(create_task(periodic()))

# async def start_background_tasks(app):
#     print("starting tasks")
#     app['tasks'] = []
#     app['tasks'].append(create_task(start_oai()))

# async def cleanup_background_tasks(app):
#     print("cleaning up tasks")
#     for t in app['tasks']:
#         log.info('shutting down openai')
#         t.cancel()
#         await t
#     log.info('done shutting down tasks')

# if False:
#     app.on_startup.append(start_background_tasks)
#     app.on_cleanup.append(cleanup_background_tasks)
# else:
#     # app.on_startup.append(start_background_tasks2)
#     pass



async def on_shutdown(app):
    # force shutdown on ctrl-c
    print('')
    kill(getpid(), 15)

app.on_shutdown.append(on_shutdown)

default_host='*'
default_port = config.get('neortc_port')

cert_file = config.get('neortc_cert','./mycert.pem')
key_file = config.get('neortc_key','./mykey.pem')

print('cert_file:', cert_file)
print('key_file:', key_file)

ssl_context = SSLContext(PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(cert_file,key_file)

print("Starting webserver port:", default_port)
web.run_app(app, host=default_host, port=default_port, ssl_context=ssl_context)
