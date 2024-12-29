import sys
import copy
import cv2 as cv
import mediapipe as mp
import numpy as np

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
    eye_width = np.abs(eye_end[0] - eye_start[0])
    relative_position = (iris_center[0] - eye_start[0]) / eye_width
    if relative_position < 0.4:
        return 'left'
    elif relative_position > 0.6:
        return 'right'
    else:
        return 'center'

def draw_gaze_arrow(image, eye_center, iris_center, direction, arrow_length):
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

                # 矢印描画
                debug_image = draw_gaze_arrow(debug_image, left_eye_info[0], left_eye_info[0], left_direction, arrow_length)
                debug_image = draw_gaze_arrow(debug_image, right_eye_info[0], right_eye_info[0], right_direction, arrow_length)

        cv.imshow('Eye Gaze Detection', debug_image)
        if cv.waitKey(1) & 0xFF == 27:  # ESCキーで終了
            break

    cap.release()
    cv.destroyAllWindows()
