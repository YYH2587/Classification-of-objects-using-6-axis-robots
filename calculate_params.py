import numpy as np

# 카메라 좌표와 월드 좌표 데이터
camera_coords = np.array([
    [1087.7, 210.55],
    [1115.4, 364.8],
    [842.41, 367.37],
    [710.37, 245.45],
    [456.66, 244.68],
    [272.7, 246.24]
])

world_coords = np.array([
    [-274.258, 367.545],
    [-287.6, 454.642],
    [-132.448, 454.824],
    [-59.659, 390.016],
    [78.439, 389.860],
    [182.002, 389.493]
])

# 선형 회귀 파라미터 계산 함수
def calculate_params(camera_coords, world_coords):
    # 확장 행렬 A 생성 (카메라 좌표 + 1)
    A = np.hstack([camera_coords, np.ones((camera_coords.shape[0], 1))])
    
    # X, Y 좌표 각각의 파라미터 계산
    params_x, _, _, _ = np.linalg.lstsq(A, world_coords[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, world_coords[:, 1], rcond=None)
    
    return params_x, params_y

# 회귀 파라미터 계산
params_x, params_y = calculate_params(camera_coords, world_coords)

# 출력
print("params_x:", params_x)
print("params_y:", params_y)
