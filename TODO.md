# TODO

## General
- [ ] refactor, cleanup dead code, address all TODOs
- [ ] PTT or Continuous Voice Mode in UI
- [ ] record all audio clips
- [ ] Test out recording while speaking
- [ ] prod, stage, debug models and sticky filters in UI
- [X] experiment with continuous mode listening UI.
- [ ] Words like 'stop" are getting eaten with silence detection?
- [X] streaming whisper... silence mode detection... barge in mode.
- [ ] app not scrolling to bottom when agent is adding history.
- [X] start local model
- [ ] escape/commands without and without exiting modality (switch contexts)
- [ ] dashboard showing all connected clients/peers and other statistics
- [ ] RAG retrieval
- [ ] notebook mode (sort of like a chat but don't prompt llm... have an escape/command mode)
- [ ] think about incremental development model... how can I use... while building
- [ ] make agent selection s
- [ ] refactor/cleanup codeticky in UI (cookie?)
- [ ] show agent selection in UI
- [ ] autoselect agent (agent priorit; prod/stage/dev)- [ ] if not secret configured not easy to tell that it's insecure.  should probably fail
- [ ] future proof server.py
- [ ] refactor UI for mobile usability
- [ ] separate cache of contexts from list of contexts; some gc of contexts... 
- [ ] Move UUIDs to LLM
- [ ] Move Peer UUID allocation to agent (away from UI)
- [ ] Add modified date to metadata
- [ ] dynamic audio/video from agent
- [ ] for "say" am I incurring multiple tts calls and just streaming audio over peers
- [ ] service client UI from agent so that I can update both sides
- [ ] Chats by date; follow openai grouping
- [ ] different clients with different contexts/peer connections
- [ ] persistence of tts text and audio files for dataset
- [ ] persistence of stt audio and text for dataset
- [ ] Make video optional (dynamic from UI) 
- [ ] some memory system.  How do I decouple interaction memories from agents?
- [ ] add stun support
- [ ] agent should have access to system stats like was stun used etc.  
- [X] multiple live models; multiple agents;
- [X] Test out webrtc's echo cancellation.
- [X] silence detection
- [X] persistence chat contexts json or sqlite? json files seems easier for now
- [X] No context… New context…
- [X] show selected context in UI
- [X] metadata listener on loaded contexts... refactor
- [X] Use summarization to generate chat display name
- [X] Create timestamp (best practices for webservices)
- [X] multiple contexts
- [X] get context history
- [X] Client can select a context
- [X] button for new context
- [X] setAudioCapture over peer connection
- [X] css for chat selection
- [X] context name and date send via events
- [X] context meta data over data channel
- [X] Make tts audio optional disable from server side (dynamic from UI) default to off
- [X] smoke test on phone; fix UI so basically useful
- [X] just disable video on clientside for now.
- [X] autostart server
- [X] use appzero certs
## My Agent
- [ ] wire up llama as llm
- [ ] Todo List is a note type
- [ ] continuous listening

## Futures
- [ ] Sound Mixing; System Sounds
- [ ] Peerconnection retry strategy… ice restart etc
- [ ] Ability to interupt streaming at anytime. When changing context.
- [ ] Different audio stream types.  system or context
- [ ] stop speaking needs to stop in flight requests to stt server
- [ ] text to speech speaking speed

---

## Thoughts

A chat is a modality
A note is a modality
A todo list is a modality
Switch between modalities
Pull in information from one modality to another (read only, copy on write)
sandbox modality
What exactly is a modality...
You have an agent that you're interacting with.  The modality is a task that allows the agent to modify a set of documents or artifacts...  Is a project or a document a better word?
agent reconnect/retry  fibonacci backoff up to max

# Unsorted
- [X] llm as context
get metadata - id, name, datecreated, sharing, model 

get context meta data for datachannel… that is what I use for the chat itself maybe just send that with the whole chat history… could optimize later if needed… 

- [X] remove peerconnection on ice failure?  no retrying for now… 





use datachannel to send text updates back
Xmove listener handling into peer speech/text

move captureAudio to datachannel
get all chat history from datachannel
multiple contexts… 
switch context
renegotiate peerconnection

speech in only respond to that peer… 
broadcast audio out peers
if I enter text… speak or don't speak
speak only if spoken to
speak always (text or speech) only to origin
speak broadcast

broadcast text everywhere

idea to experiment with vision/action/physics using a compression bottleneck


chat session metaphor
each chat session has a dialog/history
a chat session can interact with any number of agents… each agent has it's own identify/device identity (so we know which device was used for a given input…) outputs go to all agents simultaneously and can see all text inputs

av peer sessions for each device… 


chat session is an entity that exists persistently
can unthaw the chat session at any time and continue chat…
each device get it's own channel to communicate inputs…   but chat only exists at one place… multiple inputs over time… chat outputs broadcast to all clients…   what about audio output… ?   



future 

is time relevant or not… 
is device relevant or not

simplifying assumptions… each agent gets its own socket…

socket exposes request/response api 

contexts are persistent
listContexts
events for context deleted/added
connect a peer to a context… 
manage peer… renegotiate peer

peer api
datachannel for api to context…
get history
events for history changings
events for metadata changing

context could be more than just chat history text… 
timeline of events that have occurred… 

and events for managing/connecting peers to contexts

neortc api

agent api
request/response messages

request/response plumbing  

sort of want signalling server to be stupid.. blind forwards everything to agent
request/response for everything

can make rpc plumibing with completion handler/timeout

invokeAgent(agentId, method, transaction)
authenticated… 
authentication context

auth to socket
but really auth to agent 

callAgent(agentid, method, transaction)


multiple agents
openai agent

my agent… 




Neortc
create new chats
ability to select old chats and continue
filters on chat logs (filter out system)
permissions (see system messages, see tool messages)

mode change as tool
enable/disable video
enable/disable audio out
enable/disable audio in
audio out focus… just respond to speaker… respond to all
listening (guard word vs continuous listening)
voice id (only respond to me, respond to all)

what's up with delayed llm calls… they seem to queue up and are retried?  need some logging to understand
sqlite persistence or json file persistence…  maybe just json file for now… 

>Cleanup
configure logging in cfg file
Update github
Mute button
Audio in vs audio out
Think about sessions
Summarization
Persistence
Separate Threading from sessions

Get rid of aconfig
more polish on launching agent standalone or integrated
get rid of prints
agent bounces when rtc client (browser) is up connection not reestablished
Xmerge back to git… 
Xtry running on aws… 
support more than one agent
llama/llava agent… 
agent can handle more than one connection/session
peer connection not on aws so restrict ports
Think about article
Small whisper model for cpu?
optimize Ui for phone
Mute button
Interrupt
Mic button
Continual listening
P2p vs turn indicator
Home screen install
XSignal server on amazon
XTurn on amazon
XAuthl
AWS autostart python signal server
turn server with auth

Think about modalities… taking notes/research a modality… 
