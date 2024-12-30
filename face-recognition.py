import face_recognition
import cv2  # Import OpenCV
import numpy as np
import time  # 時間遅延を使うためのインポート
# Webカメラのキャプチャを開始
cap = cv2.VideoCapture(0)  # OpenCVでWebカメラをキャプチャ

# 画像を読み込み、顔の特徴値を取得する
hasegawa_image = face_recognition.load_image_file("face.jpg")
hasegawa_face_encoding = face_recognition.face_encodings(hasegawa_image)[0]

# 知っている顔の特徴値と名前
known_face_encodings = [hasegawa_face_encoding]
known_face_names = ["Hasegawa"]

while True:
    # Webカメラの1フレームを取得
    ret, frame = cap.read()
    if not ret:
        print("Webカメラのフレームを取得できませんでした")
        break

    rgb_frame = frame[:, :, ::-1]  # RGB変換

    # 顔検出
    face_locations = face_recognition.face_locations(rgb_frame)

    if face_locations:  # 顔が検出された場合のみ処理を行う
        face_encodings = []

        # 1フレームで検出した顔分ループする
        for face_location in face_locations:
            print(f"Detected face location: {face_location}")  # Debugging the face location

            # 各顔の特徴量を抽出
            try:
                face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
                face_encodings.append(face_encoding)
            except Exception as e:
                print(f"Error encoding face: {e}")  # Handling any encoding errors

        # 検出した顔それぞれについて処理を行う
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # 特徴量を比較して一致する名前を探す
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            # 顔の周りに四角を描画
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # 名前ラベルを描画
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (255, 255, 255), 2)

    # 結果を表示
    cv2.imshow('WebCam', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.3)  # 0.3秒遅延



# Webカメラとウィンドウを解放
cap.release()
cv2.destroyAllWindows()
