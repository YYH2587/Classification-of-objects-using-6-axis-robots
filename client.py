import socket
import cv2
import numpy as np
from ultralytics import YOLO

# 카메라 좌표 -> 월드 좌표 변환 함수
def camera_to_world(camera_coords):
    #calculate_param.py를 통해 구한 파라미터 값 입력
    params_x = np.array([-0.5628, 0.0117, 338.19])
    params_y = np.array([-0.0002, 0.5709, 245.72])
    A = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])
    world_x = A @ params_x
    world_y = A @ params_y
    return np.column_stack((world_x, world_y))

# 객체 탐지 및 각도 계산 함수
def calculate_angle(x_center, y_center, width, height, angle):
    short_side = min(width, height)
    long_side = max(width, height)

    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    if width < height:
        x1, y1 = x_center - short_side / 2 * cos_angle, y_center - short_side / 2 * sin_angle
        x2, y2 = x_center + short_side / 2 * cos_angle, y_center + short_side / 2 * sin_angle
    else:
        x1, y1 = x_center - short_side / 2 * sin_angle, y_center + short_side / 2 * cos_angle
        x2, y2 = x_center + short_side / 2 * sin_angle, y_center - short_side / 2 * cos_angle

    delta_x = x2 - x1
    delta_y = y2 - y1
    angle_from_x_axis = np.degrees(np.arctan2(delta_y, delta_x))

    if angle_from_x_axis < 0:
        angle_from_x_axis += 360

    angle_counterclockwise = 360 - angle_from_x_axis
    if angle_counterclockwise == 360:
        angle_counterclockwise = 0

    return angle_counterclockwise

def convert_to_rz(block_angle):
    if block_angle <= 180:
        return block_angle
    else:
        return block_angle - 180

def main():

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('192.168.1.23', 5000))

    # YOLO 모델 로드
    model = YOLO('runs/obb/yolov8-obb4/weights/best.pt')

    confidence_threshold = 0.80
    iou_threshold = 0.4

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_sent_data = None  # 마지막 전송된 데이터

    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 객체 탐지
            results = model(frame, conf=confidence_threshold, iou=iou_threshold)
            annotated_frame = results[0].plot()

            # 객체 정보 수집
            object_data = None
            if results[0].obb is not None:
                for obb in results[0].obb:
                    xywhr = obb.xywhr.detach().cpu().numpy()[0]
                    x_center, y_center, width, height, angle = xywhr

                    class_id = int(obb.cls.detach().cpu().numpy().item())
                    class_name = model.names[class_id]

                    # 좌표 변환 및 각도 계산
                    camera_coords = np.array([[x_center, y_center]])
                    world_coords = camera_to_world(camera_coords)
                    world_x, world_y = world_coords[0]

                    block_angle = calculate_angle(x_center, y_center, width, height, angle)
                    robot_rz = convert_to_rz(block_angle)

                    # 객체 데이터 생성
                    object_data = f"{world_x},{world_y},{robot_rz},{class_id}"

                    # 시각화
                    cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
                    cv2.putText(annotated_frame, f"{class_name}", 
                                (int(x_center), int(y_center) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 실시간 영상 표시
            cv2.imshow('Detection', annotated_frame)

            # ACK 처리 후 데이터 전송
            if object_data and object_data != last_sent_data:
                try:
                    sock.sendall(object_data.encode())
                    ack = sock.recv(1024).decode().strip()  # 서버 응답 대기
                    if ack == "ACK":
                        print("Server acknowledged.")
                        last_sent_data = object_data  # 마지막 전송 데이터 업데이트
                    else:
                        print("Server requested shutdown.")
                        break
                except (ConnectionResetError, ConnectionAbortedError):
                    print("Connection lost during data transfer.")
                    break

            # 'q'로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except (ConnectionResetError, ConnectionAbortedError):
            print("Connection lost.")
            break
        except Exception as e:
            print(f"Unexpected error during processing: {e}")
            break

    # 리소스 정리
    cap.release()
    cv2.destroyAllWindows()
    sock.close()
    print("Client shut down.")

if __name__ == "__main__":
    main()
