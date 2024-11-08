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

const urlParams = new URLSearchParams(window.location.search);
const peersDiv = document.querySelector("#peers")
const contextsDiv = document.querySelector("#contexts")
const remoteAudio = document.querySelector('#remote-audio')
const chatLog = document.querySelector("#chat-log")
const sendTxt = document.querySelector('#sendTxt')
const sendBtn = document.querySelector('#sendBtn')
const sockStatus = document.querySelector("#sockStatus")
const peerStatus = document.querySelector('#peerStatus')
const stopAudio = document.querySelector('#stopAudio')

function setAgent(id) {
  window.rtc.rtcConnect(id)
}

function setContext(contextStr) {
  try {
  window.rtc?.setContext(contextStr)
  } catch(e) { console.log(e)}
}

function setAudioEnabled(f) {
  window.rtc?.setAudioEnabled(f)
}

let contexts = []
let selectedContext = ''

function refreshPeers() {
  console.log('refreshPeers:', peers);
  h = '<ul>'
  for(let p in peers) {
    h += `<li><a href="." onclick="javascript:setAgent('${p}');return false;">${peers[p].displayName}</a></li>`
  }
  h += '</ul>'
  peersDiv.innerHTML = h
}

function refreshContexts() {
  console.log('getContextsResult:', contexts);
  h = '<ul>'
  h += `<li><a href="." onclick="javascript:setContext('');return false;">New</a></li>`
  for(c of contexts) {
    cl = (c.id == selectedContext)?"selected":""
    h += `<li class="${cl}"><a href="." onclick="javascript:setContext('${c.id}');return false;">${c.display}-${new Date(c.created).toLocaleString()}</a></li>`
  }
  h += '</ul>'
  contextsDiv.innerHTML = h
}

function neoRTC(url) {
  this.url = url
  this.peers = []
  this.socket = null
  this.connectStatus = false
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
      this.rtcTargetId = null
      this.channel = null
    }
  }

  this.rtcConnect = (targetId,video,audio)=>{
    this.mediaPromise.then(()=>{this._rtcConnect(targetId,video,audio)})
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
    this.socket = io.connect(url,{auth: {token: urlParams.get('token')}})

    this.socket.on("getContextsResult", (id, ctxs)=>{
      contexts = ctxs
      refreshContexts()
    })

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
    });
    
    this.socket.on("disconnect",()=>{
      this.connectStatus = false
      this.onConnectStatusChanged?.(this.connectStatus)
      this.disconnect() 
    })

  }

  this.disconnect = ()=>{
    this.rtcDisconnect()
    if (this.socket) {
      this.socket.close()
      this.socket = null
    }
  }

  this._rtcConnect = (targetId,video,audio) => {
    this.rtcTargetId = targetId

    this.socket.emit("getContexts", this.rtcTargetId)

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
        this.socket.emit("candidate", this.rtcTargetId, e.candidate);
      }
      this.onIceCandidate?.(e)      
    }
  
    this.peerConnection.onconnectionstatechange = e => {
      this.onPeerConnectionStateChange?.(this.peerConnection.connectionState)
    }
  
      try {
      this.channel = this.peerConnection.createDataChannel("chat")
      this.channel.onopen = (e)=>{
      }

      function appendLog(m) {
        console.log(m)
        h = `<div class="${m.role}"><pre><b>${m.role}:</b> ${m.content[0].text}</pre></div>`
        chatLog.innerHTML = chatLog.innerHTML + h
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
        refreshContexts()
      }
      this.channel.onmessage = (e)=>{
        console.log(e.data)
        let m = JSON.parse(e.data)

        console.log('from datachannel client:', m)
  
        if (m.t == 'replaceLog') replaceLog(m.p)
        else if (m.t == 'appendLog') appendLog(m.p)
        else if (m.t == 'onMetaDataChanged') onMetaDataChanged(m.p)
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
        this.socket.emit("offer", this.rtcTargetId, this.peerConnection.localDescription, "");
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

getStream()
  .then(getDevices)
  .then(gotDevices);

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
  console.log('urlParams:', urlParams)

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
  rtc.onPeersChanged = (p)=>{
    console.log('peers', p)
    peers = p
    refreshPeers()
  }
  rtc.connect(window.location.origin)
  //rtc.rtcConnect()
  window.rtc = rtc
  sendTxt.focus()
}

// what about keep alive... 

window.onunload = window.onbeforeunload = ()=>{
  //socket.close();
  if (window.rtc)
    rtc.disconnect()
};

