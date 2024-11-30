built on top of layers

1. web server

stateless get/post request/response api

2. signalling server

websocket/socket.io communication between potential peers used to establish a peer connection

retries forever (trivial to implement throught socket.io facilities)

3. peer connection

point to point nothing goes through the server
rtcpeerconnection between two peers used to establish a data channel established using the signalling server

retries forever (using the signalling server to reestablish the peer connection no connection to a context)

4. data channel

point to point nothing goes through the server
rtcdatachannel between two peers used to establish a bidirectional communication channel between two peers

retries forever (established once the peer connection is in place)

5. agent api over data channel

used to connect a peer connection to an agent context 