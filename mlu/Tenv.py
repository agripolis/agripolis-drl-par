################################################
# Reinforcement Learning extension for AgriPoliS
# C.Dong, 2023
################################################

import pydata_pb2 as md
import zmq

# number of input values
len_in = 67

s_socks=[]
r_socks=[]
def initzmq0(n):
    global context, s_socks, r_socks
    context = zmq.Context()
    for i in range(n):
        send_addr = "tcp://localhost:"+str(2*i+5001)
        send_socket = context.socket(zmq.PUSH)
        send_socket.connect(send_addr)
        s_socks.append(send_socket)

        recv_socket = context.socket(zmq.PULL)
        recv_addr = "tcp://0.0.0.0:"+str(2*i+5000) 
        recv_socket.bind(recv_addr)
        r_socks.append(recv_socket)



def initzmq(port_base):
    global context, send_socket, recv_socket
    context = zmq.Context()
    send_socket = context.socket(zmq.PUSH)

    send_addr = "tcp://localhost:"+str(port_base+5001) #5003"   #555"
    send_socket.connect(send_addr)

    recv_socket = context.socket(zmq.PULL)
    recv_addr = "tcp://0.0.0.0:"+str(port_base+5000) #5002" #557"
    recv_socket.bind(recv_addr)


def send_message(data,i=0):
   # Serialize the Person message to a byte string
    message_data = data.SerializeToString()

    # Send the byte string over the socket
    #send_socket.send(message_data)
    s_socks[int(i/2)].send(message_data)

def recv_message(i=0):
    data=md.RLData()
    #recv_message=recv_socket.recv()
    recv_message=r_socks[int(i/2)].recv()
    #data.ParseFromString(recv_message)
    #print(data)
    return recv_message
    #return data
      
def recv_ec(i=0):
    #recv_ec = recv_socket.recv_string()
    recv_ec = r_socks[int(i/2)].recv_string()
    #print(float(recv_ec))
    return float(recv_ec) 

def recv_closed(i=0):
    #recv = recv_socket.recv_string()
    recv = r_socks[int(i/2)].recv_string()
    #print(float(recv_ec))
    return float(recv)-1 

def send_beta(b,i=0):
    #print(b)
    msg=str(b)
    #send_socket.send_string(msg)
    s_socks[int(i/2)].send_string(msg)

def closezmq():
    global send_socket, recv_socket, context
    send_socket.close()
    recv_socket.close()
    #context.term()

def closezmq0(n):
    global context, s_socks, r_socks
    for i in range(n):
        s_socks[i].close()
        r_socks[i].close()
    context.term()
