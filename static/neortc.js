/*
# index.js
# Simple webrtc frontend

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
*/


function setCookie(name, value, days) {
  const d = new Date();
  d.setTime(d.getTime() + (days * 24 * 60 * 60 * 1000));
  const expires = "expires=" + d.toUTCString();
  document.cookie = name + "=" + value + ";" + expires + ";path=/";
}

function getCookie(name) {
  const cname = name + "=";
  const decodedCookie = decodeURIComponent(document.cookie);
  const ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) == ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(cname) == 0) {
      return c.substring(cname.length, c.length);
    }
  }
  return "";
}

function toggleTwisty() {
  const twistyContent = document.getElementById('twistyContent');
  const twistyButton = document.getElementById('twistyButton');
  if (twistyContent.style.display === 'none') {
    twistyContent.style.display = 'block';
    twistyButton.textContent = 'Show Less';
    setCookie('twistyState', 'shown', 7);
  } else {
    twistyContent.style.display = 'none';
    twistyButton.textContent = '...';
    setCookie('twistyState', 'hidden', 7);
  }
}

function togglePanel(forceClose) {
  const chatPanel = document.getElementById('chatPanel');
  if (forceClose || chatPanel.style.display === 'block') {
    chatPanel.style.display = 'none';
  } else {
    chatPanel.style.display = 'block';
    chatPanel.style.height = 'calc(100%)'; // Set height to fill the remaining space below the header
  }
}

function deleteCookie(name) {
document.cookie = name + '=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
}

function logOut() {
deleteCookie('session_token');
window.location.href = '/';
}


let enableVideo=false

let peerConnection;
const config = {
  iceServers: [
    { 
      "urls": "stun:stun.l.google.com:19302",
    },
    // { 
    //   "urls": "turn:TURN_IP?transport=tcp",
    //   "username": "TURN_USERNAME",
    //   "credential": "TURN_CREDENTIALS"
    // }
  ]
};

const peersDiv = document.querySelector("#peers")
const contextsDiv = document.querySelector("#contexts")
const remoteAudio = document.querySelector('#remote-audio')
const chatLog = document.querySelector("#chat-log")
const sendTxt = document.querySelector('#chatInput')
const sendBtn = document.querySelector('#sendButton')
const sockStatus = document.querySelector("#sockStatus")
const peerStatus = document.querySelector('#peerStatus')
const stopAudio = document.querySelector('#stopAudio')
const voiceBtn = document.querySelector('#voiceButton')
const ledIndicator = document.querySelector('#led-indicator')
const statusText = document.querySelector('#status-text')
const header = document.querySelector('#header')
const chatheader = document.querySelector('#chat-header')


function setAgent(name) {
  document.title = `Zero: ${name}`;
  header.innerHTML = `Zero: ${name}`;
  setCookie('stickyAgent',name,365)
  peers = window.rtc.peers
  id = Object.keys(peers).find(k=>peers[k].displayName == name)
  if (id) {
    if (window.rtc.channel) {
      window.rtc.channelReady.then(()=>{
        console.log('channel ready; sending getContexts:', id)
      window.rtc.channel.send(JSON.stringify({'t':'getContexts','p':id}))
      })
    }
    // TODO error handling
  }
  if (id && id != window.rtc.rtcTargetId) {
    contexts = []
    refreshPeers()
    console.log('setAgent:', id)
    window.rtc.rtcConnect(id)
  }
}

function setContext(contextStr) {
  try {
  window.rtc?.setContext(contextStr)
  togglePanel(true)
  } catch(e) { console.log(e)}
}

function setAudioEnabled(f) {
  window.rtc?.setAudioEnabled(f)
}

let contexts = []
let selectedContext = ''

function refreshPeers() {
  peers = window.rtc.peers
  console.log('refreshPeers:', peers);
  stickyAgent = getCookie('stickyAgent');
  if (stickyAgent) {
    found = Object.keys(peers).find(k=>peers[k].displayName == stickyAgent)
    if (found) {
      window.setTimeout(()=>{setAgent(stickyAgent)},1000) // TODO can I get rid of this timeout
    }
  }
  else if (Object.keys(peers).length > 0) {
    setAgent(peers[Object.keys(peers)[0]].displayName)
  }
  h = ''
  h += '<ul style="list-style-type: none; line-height: 1.5;padding-left: 5px;">'
  for(let p in peers) {
    cl = (peers[p].displayName == stickyAgent)?"selected":""
    h += `<li class="${cl}"><a href="." onclick="javascript:setAgent('${peers[p].displayName}');return false;">${peers[p].displayName}</a></li>`
  }
  if (stickyAgent && !found) {
    h += '<li class="selected">' + stickyAgent + " [OFFLINE]</li>"
  }
  h += '</ul>'

  peersDiv.innerHTML = h
}

function refreshContexts() {
  console.log('getContextsResult:', contexts);
  contexts = contexts.sort((a, b) => {
    const dateA = a.modified ? new Date(a.modified) : new Date(0);
    const dateB = b.modified ? new Date(b.modified) : new Date(0);
    return dateB - dateA;
  });
  h = ''
  h += '<ul style="list-style-type: none; line-height: 1.5;padding-left: 5px;">'
  cl = window.rtc.getStatus() == 2?"new online":"new"
  h += `<li class="${cl}"><a href="." onclick="javascript:setContext('');return false;">+New...</a></li>`
  for(c of contexts) {
    cl = (c.id == selectedContext)?"selected":""
    h += `<li class="${cl}"><a href="." title="${new Date(c.modified).toLocaleString()}" onclick="javascript:setContext('${c.id}');return false;">${c.display}</a></li>`
  }
  h += '</ul>'
  contextsDiv.innerHTML = h
}

function neoRTC(url) {
  this.url = url
  this.peers = []
  this.socket = null
  this.connectStatus = false
  this.peerStatus = false
  this.status = 0 // 0 - disconnected, 1 - connecting, 2 - connected
  this.onStatusChanged = null
  this.onConnectStatusChanged = null
  this.mediaPromise = new Promise((resolve,reject)=>{
    this.resolveMedia = resolve
    this.rejectMedia = reject
  })

  // websocket - signal server
  this.onConnect = null
  this.onDisconnect = null

  // peerconnection - rtc
  this.onPeersChanged = null
  this.onPeerConnect = null
  this.onPeerConnectionStateChange = null

  // internal housekeeping
  this._onAnswer = null
  this._onIceCandidate = null

  this._remoteDescriptionSet = false
  this._cachedIceCandidates = []

  this.rtcDisconnect = ()=>{
    if (this.peerConnection) {
      if (this.channel)
        this.channel.close()
      this.peerConnection.close()
      this.peerConnection = null
      // TODO commenting this out is probably bad
      // this.rtcTargetId = null
      this.channel = null
    }
  }

  this.rtcConnect = (targetId,video,audio)=>{
    this.rtcTargetId = targetId  // set this here since since promise will be delayed
    this.mediaPromise.then(()=>{this._rtcConnect(targetId,video,audio)})
  }

  this.getStatus = ()=>{
    if (this.connectStatus) {
      if (this.peerStatus) return 2
      else return 1
    }
    else return 0
  }
  
  this.setContext = (t)=>{
    if (this.channel)
      this.channel.send(JSON.stringify({'t':'setContext','p':t}))
  }

  this.sendText = (t)=>{
    if (this.channel)
      this.channel.send(JSON.stringify({'t':'sendText','p':t}))
  }

  this.sendStopAudio = ()=>{
    if (this.channel)
      this.channel.send(JSON.stringify({'t':'clearAudio'}))
  }

  this.setAudioEnabled = (f)=>{
    if (this.channel)
      this.channel.send(JSON.stringify({'t':'enableAudio','p':f}))
  }



  this.connect = (url)=>{
    this.disconnect() // if we're connected somewhere already disconnect

    // connect to socket...
    // get media
    // can I defer getting media...
    // can I renegotiate media
    console.log('connecting to:', url)
    // const token = document.cookie.split('; ').find(row => row.startsWith('session_token=')).split('=')[1].replace(/"/g, '');
    const token = getCookie('session_token').replace(/"/g, ''); // TODO why does the cookie have quotes
    console.log('token:', token)
    // this.socket = io.connect(url,{auth: {token: urlParams.get('token')}})
    this.socket = io.connect(url,{auth: {token: token},
      reconnection: true, // Enable reconnection
      reconnectionAttempts: Infinity, // Number of reconnection attempts before giving up
      reconnectionDelay: 1000, // Initial delay before reconnection attempt
      reconnectionDelayMax: 5000, // Maximum delay between reconnection attempts
      randomizationFactor: 0.5 // Randomization factor for reconnection delay
    })

    let self = this


    this.socket.on("peersChanged", (peers)=>{
      this.peers = peers
      console.log('peers received:', peers)
      this.onPeersChanged?.(peers)
    })
  
    this.socket.on("answer", (id, description) => {
      console.log('answer received setting remote descripiton', description)
      if (this._remoteDescriptionSet) {
        console.log('remoteDescription already set!!')
        return
      }
      if (this.peerConnection) {
        this.peerConnection.setRemoteDescription(description);
        this._remoteDescriptionSet = true
        for(let i=0; i < this._cachedIceCandidates.length; i++) {
          this.peerConnection.addIceCandidate(new RTCIceCandidate(this._cachedIceCandidates[i]))
          console.log('adding cached candidate')
        }
        this._cachedIceCandidates = []

        this._onAnswer?.(id, description)
      }
    })

    this.socket.on("candidate", (id, candidate) => {
      console.log('candidate:', candidate)
      if (this._remoteDescriptionSet)
        this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
      else
        this._cachedIceCandidates.push(candidate)
    });
    
    this.socket.on("connect",()=>{
      this.socket.emit("watcher");
      this.connectStatus = true
      this.onConnectStatusChanged?.(this.connectStatus)
      this.onStatusChanged?.(this.getStatus())
    });

    
    this.socket.on("disconnect",()=>{
      this.connectStatus = false
      this.onConnectStatusChanged?.(this.connectStatus)
      this.onStatusChanged?.(this.getStatus())
      //this.disconnect() // stop reconnection
    })

    this.socket.on("reconnect_attempt", (attemptNumber) => {
      console.log(`Reconnection attempt #${attemptNumber}`);
    });
  
    this.socket.on("reconnect", (attemptNumber) => {
      console.log(`Reconnected on attempt #${attemptNumber}`);
    });
  
    this.socket.on("reconnect_error", (error) => {
      console.error('Reconnection error:', error);
    });
  
    this.socket.on("reconnect_failed", () => {
      console.error('Reconnection failed');
    });


  }

  this.disconnect = ()=>{
    this.rtcDisconnect()
    if (this.socket) {
      this.socket.close()
      this.socket = null
    }
  }

  

  function generateGUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
      const r = (Math.random() * 16) | 0;
      const v = c === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }
  


  this._rtcConnect = (targetId,video,audio) => {
    console.log('rtcConnect:', targetId, video, audio)
    this.rtcTargetId = targetId
    this.key = generateGUID()

    // this.socket.emit("getContexts", this.rtcTargetId)

    this.rtcDisconnect()

  
    console.log("in watch handler")
    this.peerConnection = new RTCPeerConnection(config);
    this._remoteDescriptionSet = false
    this._cachedIceCandidates = []
    // adding local tracks camera
    
    
    //if (videoElement) {
    //let stream = videoElement.srcObject;
    window.stream.getTracks().forEach(track => {
      this.peerConnection.addTrack(track, stream)
      console.log('adding local tracks:', track, stream)
    });
    //}
    
    
  
    console.log('registering ontrack handler')  
      this.peerConnection.ontrack = e=>{
  
      if (e.track.kind=="audio") {
        console.log('audio stream received')
        remoteAudio.srcObject = e.streams[0]     
      }
    };

    this.peerConnection.onicecandidate = e=>{
      if (e.candidate) {
        this.socket.emit("candidate", this.rtcTargetId, e.candidate,this.key);
      }
      this.onIceCandidate?.(e)      
    }
  
    this.peerConnection.onconnectionstatechange = e => {
      console.log('connection state change:', this.peerConnection.connectionState)
      this.onPeerConnectionStateChange?.(this.peerConnection.connectionState)
      this.peerStatus = this.peerConnection.connectionState=='connected'
      this.onStatusChanged?.(this.getStatus())
    }
  
      try {
      this.channel = this.peerConnection.createDataChannel("chat")
      this.channelReady = new Promise((resolve,reject)=>{
        this.channel.onopen = (e)=>{
          console.log('data channel open')
          resolve()
        }
        // window.setTimeout(()=>{resolve()},5000)
      })

      function appendLog(m) {
        console.log(m)
        const isScrolledToBottom = chatLog.scrollHeight - chatLog.clientHeight <= chatLog.scrollTop + 1;
        const h = `<div class="${m.role}"><pre><b>${m.role}:</b> ${m.content[0].text}</pre></div>`
        chatLog.innerHTML = chatLog.innerHTML + h
        if (isScrolledToBottom) {
          chatLog.scrollTop = chatLog.scrollHeight
        }
      }
      function replaceLog(m) {
        console.log(m)
        chatLog.innerHTML = ''
        for (h of m) {
          appendLog(h)
        }
      }
      // only update context select given id
      function onMetaDataChanged(m) {
        console.log('onMetaDatachanged', m)
        selectedContext = m.id
        chatheader.innerHTML = m.display
        refreshContexts()
      }

      function onGetContextsResult(m) {
        console.log('onGetContextsResultxx', m)
        contexts = m
        refreshContexts()
      }

      this.channel.onmessage = (e)=>{
        console.log(e.data)
        let m = JSON.parse(e.data)

        console.log('from datachannel client:', m)
  
        if (m.t == 'replaceLog') replaceLog(m.p)
        else if (m.t == 'appendLog') appendLog(m.p)
        else if (m.t == 'onMetaDataChanged') onMetaDataChanged(m.p)
        else if (m.t == 'onGetContextsResult') onGetContextsResult(m.p)
        else console.log('unhandled datachannel message')
      }
      this.channel.onclose = (e)=>{
        console.log('data channel closed')
      }
    }
    catch(e) {
      console.log('data connection exception:', e)

    }

    console.log('Sending Offer')
    // .createOffer() work with audio/video
    //({offerToReceiveAudio: true,offerToReceiveVideo: false})//.createOffer()
    this.peerConnection
    .createOffer({offerToReceiveAudio: true,offerToReceiveVideo: enableVideo})//.createOffer()
      .then(sdp => this.peerConnection.setLocalDescription(sdp))
      .then(() => {
        this.socket.emit("offer", this.rtcTargetId, this.peerConnection.localDescription, "", this.key);
      });

  }


}
///// end of neoRTC

function sendPrompt() {
    window.rtc.sendText(sendTxt.value)
    sendTxt.value = ''
    sendBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>'
}

sendTxt.addEventListener("keydown",(e)=>{
  if(e.keyCode == 13 && !e.shiftKey) {
    sendPrompt()
    e.preventDefault()
  }
});

sendTxt.addEventListener("input", (e)=>{
  sendBtn.innerHTML = (sendTxt.value.length > 0)?'<i class="fa-solid fa-paper-plane"></i>':'<i class="fa-solid fa-microphone"></i>'
})

stopAudio.addEventListener('click',(e)=>{
  window.rtc.sendStopAudio()
})

let listening = false
function listen(f) {
  if (f==listening) return;
  listening = f
  if (listening) {
    sendBtn.innerHTML = '<i class="fa-solid fa-xmark"></i>'
  }
  else {
    sendBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>'
  }
  if (window.rtc.channel)
    window.rtc.channel.send(JSON.stringify({'t':'captureAudio','p':f}))
}

voiceBtn.addEventListener('click',(e)=>{
  listening = !listening
  window.rtc.channel.send(JSON.stringify({'t':'captureAudio','p':listening}))
  window.rtc?.setAudioEnabled(listening)
  voiceBtn.innerHTML = listening?'<i class="fa-solid fa-microphone-slash"></i>':'<i class="fa-brands fa-codepen"></i>'
})

sendBtn.addEventListener('pointerdown',(e)=>{
  sendBtn.setPointerCapture(e.pointerId)
  if (sendTxt.value.length == 0)
    listen(true)
})
sendBtn.addEventListener('pointerup',(e)=>{
  sendBtn.releasePointerCapture(e.pointerId)
  if (sendTxt.value.length == 0)
    listen(false)
  else sendPrompt()
})

// Handle HTML integration

//Get camera and microphone
const videoElement = document.querySelector("#local_video");
const remoteVideo = document.querySelector("#remote_video")
const audioSelect = document.querySelector("select#audioSource");
const videoSelect = document.querySelector("select#videoSource");

audioSelect.onchange = getStream;
videoSelect.onchange = getStream;



getDevices().then(info=>{
  console.log('deviceinfo:',info)
})

function getDevices() {
  return navigator.mediaDevices.enumerateDevices();
}

function gotDevices(deviceInfos) {
  window.deviceInfos = deviceInfos;
  for (const deviceInfo of deviceInfos) {
    const option = document.createElement("option");
    option.value = deviceInfo.deviceId;
    if (deviceInfo.kind === "audioinput") {
      option.text = deviceInfo.label || `Microphone ${audioSelect.length + 1}`;
      audioSelect.appendChild(option);
    } else if (deviceInfo.kind === "videoinput") {
      option.text = deviceInfo.label || `Camera ${videoSelect.length + 1}`;
      videoSelect.appendChild(option);
    }
  }
}

function getStream() {
  if (window.stream) {
    // shut down any running tracks
    window.stream.getTracks().forEach(track => {
      track.stop();
    });
  }
  const audioSource = audioSelect.value;
  const videoSource = videoSelect.value;
  let constraints = {
    audio: { deviceId: audioSource ? { exact: audioSource } : undefined }//,
    //video: { deviceId: videoSource ? { exact: videoSource } : undefined }
  };
  if (enableVideo) {
    constraints = {
      audio: { deviceId: audioSource ? { exact: audioSource } : undefined },
      video: { deviceId: videoSource ? { exact: videoSource } : undefined }
    };
  }
  return navigator.mediaDevices
    .getUserMedia(constraints)
    .then(gotStream)
    .catch(handleError);
}

function gotStream(stream) {
  window.stream = stream; // is this needed?  just a global reference
  audioSelect.selectedIndex = [...audioSelect.options].findIndex(
    option => option.text === stream.getAudioTracks()[0].label
  );
  if (enableVideo) {
    videoSelect.selectedIndex = [...videoSelect.options].findIndex(
      option => option.text === stream.getVideoTracks()[0].label
    );
    videoElement.srcObject = stream;
  }
  // console.log('urlParams:', urlParams)

  window.rtc.resolveMedia()
}

function handleError(error) {
  console.error("Error: ", error);
}

window.onload = ()=>{
  document.querySelector('#version').innerHTML = '3';
  rtc = new neoRTC()
  rtc.onConnectStatusChanged = (f)=>{
    sockStatus.innerHTML = f?'connected':'disconnected'
  }
  rtc.onPeerConnectionStateChange = (f)=>{
    peerStatus.innerHTML = f
  }
  rtc.onStatusChanged = (s)=>{
    console.log('status:', s)
    cText = ['Off','Off','On']
    cColor = ['red','yellow','green']
    ledIndicator.style.backgroundColor = cColor[s]
    statusText.innerHTML = cText[s]

    if (s != 2) {
      contexts = []
      refreshContexts()
    }

    const elements = document.querySelectorAll('.new')
    // elements.forEach(e=>e.style.display = (s==2)?'block':'none')
    if (s==2) {
      elements.forEach(e=>e.classList.add('online'))
    }
    else {
      elements.forEach(e=>e.classList.remove('online'))
    }
  }
  rtc.onPeersChanged = (p)=>{
    console.log('peers', p)
    peers = p
    refreshPeers()
  }
  rtc.connect(window.location.origin)
  //rtc.rtcConnect()
  window.rtc = rtc
  getStream()
  .then(getDevices)
  .then(gotDevices);

    // window.onload = function() {
  const twistyState = getCookie('twistyState');
  const twistyContent = document.getElementById('twistyContent');
  const twistyButton = document.getElementById('twistyButton');
  if (twistyState === 'shown') {
    twistyContent.style.display = 'block';
    twistyButton.textContent = 'Show Less';
  } else {
    twistyContent.style.display = 'none';
    twistyButton.textContent = '...';
  }
    // }

  sendTxt.focus()
}

// what about keep alive... 

window.onunload = window.onbeforeunload = ()=>{
  //socket.close();
  if (window.rtc)
    rtc.disconnect()
};

