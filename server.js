
const webSocketsServerPort = 8000; 
const webSocketServer = require('websocket').server; 
const http = require('http'); 
const { CLIENT_RENEG_LIMIT, checkServerIdentity } = require('tls');

// Create server 
const server = http.createServer(); 
server.listen(webSocketsServerPort); 
const wsServer = new webSocketServer({ 
    httpServer: server
}); 

const clients = {}; 

// Generate UUID for users 
const getUUID = () => { 
    const s4 = () => Math.floor((1+ Math.random()) * 0x10000).toString(16).substring(1); 
    return s4() + s4() + '-' + s4(); 
}; 

// Create connection 
wsServer.on('request', function(request) {
    var userID = getUUID();
    console.log((new Date()) + ' Recieved a new connection from origin ' + request.origin + '.');
    const connection = request.accept(null, request.origin);
    clients[userID] = connection;
    console.log('connected: ' + userID + ' in ' + Object.getOwnPropertyNames(clients))
  });

// Client logs out of webpage  
