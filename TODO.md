# TODO

cleanup plugin naming conventions async/sync tts stt etc...

-[X] bargeIn is not working
stt.clearAudio is not quite what I want... I want to cancel all submtted llm streaming requests to stt... 

Streaming support
on prompt text create a "generation id" that - [ ] break out css and organize
is used to track the response (text or audio)
message or start generating using same "generation id"
append to log entry... same "generation" id
message for stop generating same "generation" id
be able to stop/cancel generation
audio output if enable has same generation id and can be cancelled
cancels all upcoming/queued transactions.
assumes request/response... 

# simplifying assumption only one entry can be open at a time.  rest are queued is this really
# a problem?
# on client side.  add a div with id on open with animated distractor at bottom on write Entry update div with correct id.
# for now log errors if I get a openEntry when there is already an entry open... 
# prompts are updated on close with full accumulated entry
# if a prompt comes in while an entry is open... queued it up... 
# I need to look at bargeIn again to see how I want to wire that up.
# bargeIn should at a minimum stop tts output and allow new audio input to be taken in.

openEntry(id,role)
writeEntry(id,)
closeEntry(id,reason) # done, cancelled

what if initiated by agent... not a response to a prompt

is a trigger a kind of prompt... alarm or external event is like a trigger




## General
- [ ] better output handling for code and structued output
- [ ] better UI handling for graphical things like image generation
- [ ] try app generation and code generation
- [ ] better status reporting in client when doing large model loads
- [ ] record at max frequency.  and record at whisper frequency for debugging
- [ ] break out css and organize
- [ ] Add stock lookup, weather lookup, time lookup
- [ ] hook up rag
- [ ] accumlate bookmarks... x, browser, youtube, pocket... 
- [ ] hook up recipes... chat with my recipes
- [ ] Some distractor when agent is thinking
- [ ] backup models... 
- [ ] try to deal with background audio... like news in the background
- [ ] notebooklm do a paper review... 
- [ ] wire up 'stop' to stop generating and clear audio.
- [X] Streaming tts on period or nl….
- [X] filter out system prompt by default
- [ ] experiment with making audio dialog better.  be able to tweak values from UI
- [ ] clean up logging; color coding
- [ ] context list has a bit of latency now
- [ ] refactor/cleanup client code
- [ ] llama generation very slow
- [ ] llama agent isn't exiting properly with crtl-c
- [ ] add ability to delete context from UI
- [ ] add ability to edit past input from UI (fork or edit)
- [ ] add ability to share context from UI
- [ ] add ability to regenerate content generation back in time
- [ ] add ability to rename context from UI
- [ ] exception handling if config file has errors
- [ ] Better UI handling when not connected to agent
- [ ] better icon/splash screen
- [ ] search chat list
- [ ] better auth if no session token on any screen redirect to auth
- [ ] can I auto connect to agent?
- [ ] switch back to llama by default
- [ ] try to make a notebook mode
- [ ] play around with rag retrieval
- [ ] Could perhaps use a generation id to better cancel future queued audio tts given stt input
- [ ] record all audio clips
- [ ] Does audio recording need another thread?
- [ ] refactor, cleanup dead code, address all TODOs
- [ ] clean up UI for AVM (Advanced Voice Mode)
- [ ] clean up UI for phone and tablet
- [ ] clean up logging
- [ ] PTT or Continuous Voice Mode in UI
- [ ] aggregate speech segments to send to llm
- [ ] Indicator that agent is connected; heartbeat
- [X] Client reconnecting
- [ ] Agent reconnecting
- [ ] Current peer reconnecting
- [ ] Access logging
- [ ] Access metering
- [ ] Better security for agent.
- [ ] if angle brackets are in the output they are not escaped properly in the log
- [ ] When asked to stop or be quiet don't continue to talk
- [ ] Be able to configure properties from the client kv pairs.
- [ ] Test out recording while speaking
- [ ] prod, stage, debug models and sticky filters in UI
- [X] Words like 'stop" are getting eaten with silence detection?
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
- [ ] background voice mode on android (will I need a service?)
- [X] Add modified to context metadata
- [X] sort by modified date
- [X] logout button
- [X] add streaming support for llama
- [X] add streaming support to tts output
- [X] move status to top
- [X] handle peer connection state in status
- [X] move recordings into a directory 
- [X] Some auth so that I can get rid of the token and make the home page app work
- [X] when adding page to homescreen can I hide urlbar?
- [X] auto dismiss panel when making a selection- [X] experiment with continuous mode listening UI.
- [X] Sometimes bargein is being triggered given audio output.
- [X] streaming whisper... silence mode detection... barge in mode.
- [X] app not scrolling to bottom when agent is adding history.
- [X] start local model
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
- [X] agent id is not stable across restarts - make sticky work
- [X] hide crap in panel with a twisty and persist in cookie (default hidden)
- [X] show selected context/chat in UI
- [X] show selected context at top of chat and update when metadata changes
- [X] why does clicking on 'new untitled' open panel?
- [X] move getting contexts and metadata to datachannel
- [X] clean up agent list in UI css and current selected one.
- [X] move logout above 'junk drawer'
- [X] can I get rid of the extra temporary untitled documents and the slow garbage collection process
- [X] context list not refreshed when disconnected
- [X] switching contexts should likely stop audio output
- [X] only send full context list at beginning and then send deltas (add, delete, modify)
- [X] race condition on chrome mobile.  data channel not open yet in setContext
- [X] can I stream output from the llm to the client?
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
