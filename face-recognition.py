import cv2
import face_recognition
import numpy as np

# 学習データの顔画像（ここでは顔画像 'face.jpg' と 'face1.jpg' を使用）
train_imgs = [face_recognition.load_image_file("face.jpg"), face_recognition.load_image_file("face1.jpg")]
train_img_encodings = [face_recognition.face_encodings(img)[0] for img in train_imgs]

# 学習データの名前
known_face_names = ["Person 1", "Person 2"]

# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(0)  # OpenCVでWebカメラをキャプチャ

while True:
    # 映像のフレームをキャプチャ
    ret, frame = cap.read()

    # キャプチャしたフレームをRGB形式に変換（face_recognitionはRGBを使用）
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # 映像内の顔を検出
    face_locations = face_recognition.face_locations(rgb_frame)

    # 顔が検出された場合にのみ処理を行う
    if face_locations:
        # 顔の特徴量（エンコーディング）を取得
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # 検出した顔ごとに認証を行う
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 学習データとのユークリッド距離を計算
            dists = face_recognition.face_distance(train_img_encodings, face_encoding)

            # 類似度が0.40以下であれば一致と判定
            name = "Unknown"
            min_dist = min(dists)

            if min_dist < 0.40:  # 類似度が0.40以下の場合に一致と判定
                match_index = np.argmin(dists)  # 最小距離のインデックスを取得
                name = known_face_names[match_index]

            # 顔の周りに四角形を描画
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 名前を顔の上に表示
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, top - 6), font, 0.5, (255, 255, 255), 1)

    # フレームを表示
    cv2.imshow('Face Recognition', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Webカメラを解放
cap.release()
cv2.destroyAllWindows()
