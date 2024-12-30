import sys
import copy
import numpy as np
import cv2 as cv
import mediapipe as mp
import time  # 時間遅延を使うためのインポート

# 矢印の描画パラメータ
arrow_length = 50
arrow_color = (0, 255, 0)  # 矢印の色を設定（BGR形式）

# モデルロード
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

def calc_min_enc_losingCircle(landmark_list):
    center, radius = cv.minEnclosingCircle(np.array(landmark_list))
    center = (int(center[0]), int(center[1]))
    radius = int(radius)
    return center, radius

def calc_iris_min_enc_losingCircle(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append((landmark_x, landmark_y))
    left_eye_points = [landmark_point[468], landmark_point[469], landmark_point[470], landmark_point[471], landmark_point[472]]
    right_eye_points = [landmark_point[473], landmark_point[474], landmark_point[475], landmark_point[476], landmark_point[477]]
    left_eye_info = calc_min_enc_losingCircle(left_eye_points)
    right_eye_info = calc_min_enc_losingCircle(right_eye_points)
    return left_eye_info, right_eye_info

def get_eye_direction(eye_start, eye_end, iris_center):
    # 目の水平方向の長さを計算
    eye_width = np.abs(eye_end[0] - eye_start[0])
    
    # 虹彩の中心が目の水平方向のどの位置にあるかを計算
    relative_position = (iris_center[0] - eye_start[0]) / eye_width
    
    # 両目の位置が非常に近い場合（目が正面を向いている場合）は"center"
    if abs(eye_end[0] - eye_start[0]) < 10:  # Threshold for straight-on alignment
        return 'center'
    
    # 虹彩の位置に基づいて方向を判断
    if relative_position < 0.4:
        return 'left'
    elif relative_position > 0.6:
        return 'right'
    else:
        return 'center'


def is_centered(left_direction, right_direction):
    # 両目ともにcenterならcenterとして判定
    return left_direction == 'center' and right_direction == 'center'

def draw_gaze_arrow(image, eye_center, iris_center, direction, arrow_length):
    """
    目がどちらを向いているかを示す矢印を描画する関数

    Parameters:
    - image: 画像
    - eye_center: 目の中心の座標
    - iris_center: 虹彩の中心の座標
    - direction: 'left', 'right', or 'center'
    - length: 矢印の長さ

    Returns:
    - image: 矢印が描画された画像
    """
    if direction == 'left':
        end_point = (eye_center[0] - arrow_length, eye_center[1])
    elif direction == 'right':
        end_point = (eye_center[0] + arrow_length, eye_center[1])
    else:  # center
        end_point = iris_center
    cv.arrowedLine(image, eye_center, end_point, (255, 0, 0), 2, tipLength=0.3)
    return image

if __name__ == '__main__':
    cap = cv.VideoCapture(0)  # Webカメラをキャプチャ

    prev_left_direction = None
    prev_right_direction = None

    while True:
        ret, image = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(image)
        image_width, image_height = image.shape[1], image.shape[0]

        # 検出実施
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # 描画
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_info, right_eye_info = calc_iris_min_enc_losingCircle(image, face_landmarks)
                left_direction = get_eye_direction(left_eye_info[0], right_eye_info[0], left_eye_info[0])
                right_direction = get_eye_direction(left_eye_info[0], right_eye_info[0], right_eye_info[0])

                # 目線方向の変化ログを出力
                if not is_centered(left_direction, right_direction):
                    if left_direction != prev_left_direction or right_direction != prev_right_direction:
                        print(f"Left Eye Direction: {left_direction}, Right Eye Direction: {right_direction}")
                        prev_left_direction = left_direction
                        prev_right_direction = right_direction
                else:
                    print(f"Both Eyes Centered: Left Eye Direction: {left_direction}, Right Eye Direction: {right_direction}")

                # 矢印描画
                debug_image = draw_gaze_arrow(debug_image, left_eye_info[0], left_eye_info[0], left_direction, arrow_length)
                debug_image = draw_gaze_arrow(debug_image, right_eye_info[0], right_eye_info[0], right_direction, arrow_length)

        # 画面に結果を表示
        cv.imshow("Gaze Direction", debug_image)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()