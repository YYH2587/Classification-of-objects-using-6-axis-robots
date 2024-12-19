#!/usr/bin/python
# -*- coding: utf-8 -*- 

## 1. 초기 설정 #######################################
from i611_MCS import *
from i611_extend import *
from rbsys import *
from i611_common import *
from i611_io import *
from i611shm import *
from teachdata import *
import socket
import time

# 소켓 서버 초기화
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('192.168.1.23', 5000))  # 서버의 IP 주소와 포트 설정
server_socket.listen(0)
print("Waiting for a connection from the computer...")
sock, addr = server_socket.accept()
print("Connected")

# 디지털 출력 제어 함수
def out(index, num):  # Digital Output 제어
    dout(index, str(num))  # index 포트부터 연속적인 포트 제어, num 에 입력받은 숫자로 비트 제어
    time.sleep(0.2)

def main():
    ## 2. 초기 설정 ####################################
    rb = i611Robot()
    _BASE = Base()
    rb.open()
    IOinit(rb)
    
    m = MotionParam(jnt_speed=20, lin_speed=20, pose_speed=15, overlap=30)
    rb.motionparam(m)

    def close_gripper():
        out(48, 1)
        out(48, 0)

    def open_gripper():
        out(50, 1)
        out(50, 0)

    start_joint = Joint(90, -36.868, -113.585, 0, -29.130, 0)
    place_p0 = Position(x=-200, y=-369, z=150, Rz=-105, Ry=0, Rx=180, posture=6)
    place_p1 = Position(x=-302, y=-369, z=150, Rz=-117, Ry=0, Rx=180, posture=6)
    place_p2 = Position(x=-77, y=-369, z=150, Rz=-86.4, Ry=0, Rx=180, posture=6)
    cnt_0, cnt_1, cnt_2 = 0, 0, 0

    open_gripper()
    rb.home()
    rb.move(start_joint)

    while True:
        try:
            sock.settimeout(15)
            data = sock.recv(1024)
            if not data:
                print("Client disconnected or no data received. Moving robot to home position and shutting down...")
                break

            message = data.decode().strip()
            objects_data = message.split(";")

            for obj in objects_data:
                if obj:
                    world_x, world_y, robot_rz, class_id = map(float, obj.split(','))
                    class_id = int(class_id)

                    print("Received Data - X: {}, Y: {}, RZ: {}, Class ID: {}".format(world_x, world_y, robot_rz, class_id))

                    pick_p = Position(x=world_x, y=world_y, z=100, Rz=robot_rz, Ry=0, Rx=180, posture=6)

                    rb.move(pick_p.offset(dz=50))
                    rb.line(pick_p)

                    close_gripper()
                    rb.line(pick_p.offset(dz=50))

                    if class_id == 0:
                        rb.move(place_p0)
                        rb.line(place_p0.offset(dz=-50+(cnt_0*15)))
                        open_gripper()
                        rb.line(place_p0)
                        cnt_0 += 1
                    elif class_id == 1:
                        rb.move(place_p1)
                        rb.line(place_p1.offset(dz=-50+(cnt_1*15)))
                        open_gripper()
                        rb.line(place_p1)
                        cnt_1 += 1
                    elif class_id == 2:
                        rb.move(place_p2)
                        rb.line(place_p2.offset(dz=-50+(cnt_2*15)))
                        open_gripper()
                        rb.line(place_p2)
                        cnt_2 += 1
                    

            sock.sendall(b"ACK\n")  # ACK 전송

        except (socket.timeout):
            print("Client disconnected or no data received. Moving robot to home position and shutting down...")
            break

        except Exception as e:
            print("Error processing data: {}".format(e))
            continue

    rb.home()  # 로봇 팔 초기 위치로 이동
    sock.close()
    rb.close()
    print("Server shut down.")

if __name__ == "__main__":
    main()
