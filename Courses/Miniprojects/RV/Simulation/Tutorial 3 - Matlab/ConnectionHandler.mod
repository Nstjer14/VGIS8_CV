MODULE ConnectionHandler
    
LOCAL VAR socketdev socketServer;
LOCAL VAR socketdev socketClient;

PROC CONH_EstablishConnection(string ipAddress, num portNumber)

    SocketCreate socketServer;
    SocketBind socketServer, ipAddress, portNumber;
    SocketListen socketServer;
    SocketAccept socketServer,socketClient,\ClientAddress:=ipAddress,\Time:=WAIT_MAX;
     
ENDPROC

PROC CONH_RecieveData(INOUT string receive_string)
    
    SocketReceive socketClient \Str := receive_string, \Time:=WAIT_MAX;

ENDPROC

PROC CONH_SendData(string send_string)
    
    SocketSend socketClient \Str := send_string;
    
ENDPROC

ENDMODULE