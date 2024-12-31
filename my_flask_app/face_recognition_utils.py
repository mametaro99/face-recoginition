import face_recognition
import numpy as np
import cv2
import os

def recognize_face_from_camera(users):
    """
    引数として渡されたユーザー情報（名前と顔写真パス）を用いて顔認証を行う。

    Args:
        users (list): 辞書形式のユーザー情報のリスト。例: [{'name': 'user1', 'face_image': 'path/to/image.jpg'}, ...]

    Returns:
        str: 一致したユーザー名（認証失敗時は None）。
    """
    known_face_encodings = []
    known_face_names = []

    for user in users:
        if user['face_image']:
            try:
                img = face_recognition.load_image_file(user['face_image'])
                encoding = face_recognition.face_encodings(img)[0]
                known_face_encodings.append(encoding)
                known_face_names.append(user['name'])
            except Exception as e:
                print(f"Error processing image for {user['name']}: {e}")
                continue

    cap = cv2.VideoCapture(0)  # Webカメラを起動

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])  # RGB形式に変換
        face_locations = face_recognition.face_locations(rgb_frame)

        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                dists = face_recognition.face_distance(known_face_encodings, face_encoding)
                min_dist = min(dists)

                if min_dist < 0.40:  # 類似度が0.40以下なら一致
                    match_index = np.argmin(dists)
                    name = known_face_names[match_index]

                    # 顔の位置に名前を描画
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    cap.release()
                    cv2.destroyAllWindows()
                    return name

        # フレームを表示
        cv2.imshow('Face Recognition', frame)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None
