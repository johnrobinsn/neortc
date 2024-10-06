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
// console.log('url:', window.location.origin+'?token='+encodeURIComponent(urlParams.get('token')))
const peers = document.querySelector("#peers")
const talkBtn = document.querySelector("#talk-button")
const remoteAudio = document.querySelector('#remote-audio')
const chatLog = document.querySelector("#chat-log")
const sendTxt = document.querySelector('#sendTxt')
const sendBtn = document.querySelector('#sendBtn')
const sockStatus = document.querySelector("#sockStatus")
const peerStatus = document.querySelector('#peerStatus')


//const socket = io.connect(window.location.origin+'?token='+encodeURIComponent(urlParams.get('token')))

//socket = io.connect(window.location.origin)

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

  this.rtcDisconnect = ()=>{
    if (this.peerConnection) {
      this.peerConnection.close()
      this.peerConnection = null
      this.rtcTargetId = null
    }
  }

  this.rtcConnect = (targetId,video,audio)=>{
    this.mediaPromise.then((targetId,video,audio)=>{this._rtcConnect(targetId,video,audio)})
  }

  this.sendText = (t)=>{
    this.socket.emit('sendText', this.rtcTargetId, t)
  }

  this.connect = (url)=>{
    this.disconnect() // if we're connected somewhere already disconnect

    // connect to socket...
    // get media
    // can I defer getting media...
    // can I renegotiate media
    this.socket = io.connect(url,{auth: {token: urlParams.get('token')}})

    this.socket.on("peersChanged", (peers)=>{
      this.peers = peers
      console.log('peers received:', peers)
      this.onPeersChanged?.(peers)
    })
  
    this.socket.on("answer", (id, description) => {
      //peerConnections[id].setRemoteDescription(description);
      if (this.peerConnection) {
        this.peerConnection.setRemoteDescription(description);
        this._onAnswer?.(id, description)
      }
    })

    this.socket.on("candidate", (id, candidate) => {
      console.log('candidate:', candidate)
      //peerConnections[id].addIceCandidate(new RTCIceCandidate(canwindow.location.origindate));
      this.peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
    });

	  // internal housekeeping... 
    // this.socket.on("disconnectPeer", id => {
    //   console.log('disconnectPeer', id)
    //   this.peerConnection.close();
    //   this.onDisconnectPeer?.(this.socket.id)
    //   //peerConnections[id].close();
    //   //delete peerConnections[id];
    // });
    
    this.socket.on("connect",()=>{
      this.socket.emit("watcher");
      this.connectStatus = true
      this.onConnectStatusChanged?.(this.connectStatus)
      //window.rtc.rtcConnect()
    });
    
    this.socket.on("disconnect",()=>{
      //sockStatus.innerHTML = 'disconnected'
      //this.onDisconnect?.()
      this.connectStatus = false
      this.onConnectStatusChanged?.(this.connectStatus)
      this.disconnect() 
    })

    // check to see if socket is already disconnected prior to registering
    // ondisconnect handler
    // console.log('socket status:', this.socket)
    // if (this.socket.disconnected) {
    //   this.connectStatus = false
    //   this.onConnectStatusChanged?.(this.connectStatus)
    //   this.disconnect() 
    // }

    this.socket.on("onMessage", (id, m)=>{
      console.log('client:', m)
  
      h = `<div class="${m.role}"><pre><b>${m.role}:</b> ${m.content[0].text}</pre></div>`
      chatLog.innerHTML = chatLog.innerHTML + h
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
    this.rtcDisconnect()

    this.rtcTargetId = targetId
  
    console.log("in watch handler")
    this.peerConnection = new RTCPeerConnection(config);
    //peerConnections[id] = peerConnection;
  
    // adding local tracks camera
    if (videoElement) {
    let stream = videoElement.srcObject;
    window.stream.getTracks().forEach(track => {
      this.peerConnection.addTrack(track, stream)
      console.log('adding local tracks:', track, stream)
    });
    }
  
    console.log('registering ontrack handler')  
    //peerConnections[id].ontrack = e=>{
      this.peerConnection.ontrack = e=>{
      //console.log("on track received trying to loop back", e)
  //    remoteVideo.srcObject = e.streams[0]
      //peerConnection.addTrack(e.streams[0])
    //   if (e.streams.length > 0) {
    //   let stream = new MediaStream()//e.streams[0];//videoElement.srcObject;
    //   e.streams[0].getTracks().forEach(track =>{
    //     // todo right now really doing local video... should I switch to remote video
    //     // issue is I can't figure out how to reflect video when the receiver is javascript rather
    //     // than python
    //     if (track.kind=="video") {
    //       //peerConnection.addTrack(track, stream)
    //       //console.log('adding remote tracks', track, stream)
    //     }
    //     else if (track.kind=="audio") {
    //       remoteAudio.srcObject = e.streams[0]
    //     }
    //   });    
    // }
    // else console.log('no streams in track')
  
      if (e.track.kind=="audio") {
        console.log('audio stream received')
        remoteAudio.srcObject = e.streams[0]
        // remoteAudio.setAttribute('playsinline', true)
        // remoteAudio.play()

        // const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        // const source = audioContext.createMediaStreamSource(e.streams[0]);
        // const destination = audioContext.createMediaStreamDestination();
        // source.connect(destination);        

      }
      // else if (e.track.kind=="video") {
      //   remoteVideo.srcObject = e.streams[0]
      // }
      // if (e.candidate) {
      //   this.socket.emit("candidate", this.rtcTargetId, e.candidate);
      // }
      // this.onIceCandidate?.(e)
    };

    this.peerConnection.onicecandidate = e=>{
      if (e.candidate) {
        this.socket.emit("candidate", this.rtcTargetId, e.candidate);
      }
      this.onIceCandidate?.(e)      
    }
  
    this.peerConnection.onconnectionstatechange = e => {
      //peerStatus.innerHTML = this.peerConnection.connectionState
      this.onPeerConnectionStateChange?.(this.peerConnection.connectionState)
    }
  
    this.peerConnection
    .createOffer()//({offerToReceiveAudio: true,offerToReceiveVideo: false})//.createOffer()
      .then(sdp => this.peerConnection.setLocalDescription(sdp))
      .then(() => {
        this.socket.emit("offer", this.rtcTargetId, this.peerConnection.localDescription);
      });
  }


}
///// end of neoRTC

function sendPrompt() {
    window.rtc.sendText(sendTxt.value)
    chatLog.innerHTML = chatLog.innerHTML + sendTxt.value
    sendTxt.value = ''
    sendBtn.innerHTML = '<i class="fa-solid fa-microphone"></i>'
}

// sendBtn.addEventListener('click',(e)=>{
//   sendPrompt()
// })

sendTxt.addEventListener("keydown",(e)=>{
  if(e.keyCode == 13 && !e.shiftKey) {
    sendPrompt()
    e.preventDefault()
  }
});

sendTxt.addEventListener("input", (e)=>{
  sendBtn.innerHTML = (sendTxt.value.length > 0)?'<i class="fa-solid fa-paper-plane"></i>':'<i class="fa-solid fa-microphone"></i>'
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
  // todo
  window.rtc.socket.emit('captureAudio',window.rtc.rtcTargetId,f)
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

// function watch(id) {
//   watching = id
//   watchingdiv.innerHTML = `watching: ${bcasters[id].displayName}`
//   socket.emit("watch",id);
// }

// socket.on("peersChanged", (b)=>{

//   bcasters = b
//   console.log('peers received')
//   peers.innerHTML = ''
//   let found = false
//   for (var k in b) {
//     peers.insertAdjacentHTML('beforeend',`<a href="" onclick="watch('${k}');return false">${b[k].displayName}</a><br>`)
//     if (watching == k)
//       found = true
//   }
//   // if (!found) {
//   //   let k = Object.keys(b)
//   //   if (k.length > 0)
//   //     setTimeout(()=>watch(k[0]),0)
//   // }
// });

// // socket.on("answer", (id, description) => {
// //   //peerConnections[id].setRemoteDescription(description);
// //   peerConnection.setRemoteDescription(description);

// // });

// //socket.on("watcher", id => {
// function watch(id) {
//   target = id

//   console.log("in watch handler")
//   peerConnection = new RTCPeerConnection(config);
//   //peerConnections[id] = peerConnection;

//   // adding local tracks camera
//   if (videoElement) {
//   let stream = videoElement.srcObject;
//   stream.getTracks().forEach(track => {
//     peerConnection.addTrack(track, stream)
//     console.log('adding local tracks:', track, stream)
//   });
//   }

//   console.log('registering ontrack handler')  
//   //peerConnections[id].ontrack = e=>{
//     peerConnection.ontrack = e=>{
//     //console.log("on track received trying to loop back", e)
// //    remoteVideo.srcObject = e.streams[0]
//     //peerConnection.addTrack(e.streams[0])
//   //   if (e.streams.length > 0) {
//   //   let stream = new MediaStream()//e.streams[0];//videoElement.srcObject;
//   //   e.streams[0].getTracks().forEach(track =>{
//   //     // todo right now really doing local video... should I switch to remote video
//   //     // issue is I can't figure out how to reflect video when the receiver is javascript rather
//   //     // than python
//   //     if (track.kind=="video") {
//   //       //peerConnection.addTrack(track, stream)
//   //       //console.log('adding remote tracks', track, stream)
//   //     }
//   //     else if (track.kind=="audio") {
//   //       remoteAudio.srcObject = e.streams[0]
//   //     }
//   //   });    
//   // }
//   // else console.log('no streams in track')

//     if (e.track.kind=="audio") {
//       console.log('audio stream received')
//       remoteAudio.srcObject = e.streams[0]
//     }
//     // else if (e.track.kind=="video") {
//     //   remoteVideo.srcObject = e.streams[0]
//     // }

//   }  

//   peerConnection.onicecandidate = event => {
//     if (event.candidate) {
//       socket.emit("candidate", id, event.candidate);
//     }
//   };

//   peerConnection.onconnectionstatechange = e => {
//     peerStatus.innerHTML = peerConnection.connectionState
//   }

//   peerConnection
//   .createOffer()//.createOffer({offerToReceiveAudio: true,offerToReceiveVideo: true})
//     .then(sdp => peerConnection.setLocalDescription(sdp))
//     .then(() => {
//       socket.emit("offer", id, peerConnection.localDescription);
//     });
// }
// //});

// socket.on("onMessage", (id, m)=>{
//   console.log('client:', m)

//   h = `<div class="${m.role}"><b>${m.role}: </b> ${m.content[0].text}</div>`
//   chatLog.innerHTML = chatLog.innerHTML + h
// })

// socket.on("candidate", (id, candidate) => {
//   console.log('candidate:', candidate)
//   //peerConnections[id].addIceCandidate(new RTCIceCandidate(canwindow.location.origindate));
//   peerConnection.addIceCandidate(new RTCIceCandidate(candidate));
// });

// socket.on("disconnectPeer", id => {
//   console.log('disconnectPeer', id)
//   peerConnection.close();
//   //peerConnections[id].close();
//   //delete peerConnections[id];
// });

// socket.on("connect", () => {
//   sockStatus.innerHTML = 'connected'
//   socket.emit("watcher");

//   // let displayName = urlParams.get('name')?urlParams.get('name'):socket.id
//   // socket.emit("broadcaster", {'displayName':displayName});  
// });


// socket.on("disconnect", ()=> {
//   sockStatus.innerHTML = 'disconnected'
// })

// window.onunload = window.onbeforeunload = () => {
//   socket.close();
// };


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
  const constraints = {
    audio: { deviceId: audioSource ? { exact: audioSource } : undefined },
    video: { deviceId: videoSource ? { exact: videoSource } : undefined }
  };
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
  videoSelect.selectedIndex = [...videoSelect.options].findIndex(
    option => option.text === stream.getVideoTracks()[0].label
  );
  videoElement.srcObject = stream;
  console.log('urlParams:', urlParams)
  //let displayName = urlParams.get('name')?urlParams.get('name'):socket.id
  //socket.emit("broadcaster", {'displayName':displayName});
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
  rtc.connect(window.location.origin)
  rtc.rtcConnect()
  window.rtc = rtc
}

// what about keep alive... 

window.onunload = window.onbeforeunload = ()=>{
  //socket.close();
  if (window.rtc)
    rtc.disconnect()
};
// socket.on("connect", ()=> {

// })
