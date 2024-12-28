import face_recognition
from PIL import Image, ImageDraw

# 画像を読み込む
load_image = face_recognition.load_image_file("images/faces.jpg")

# 認識させたい画像から顔検出する
face_locations = face_recognition.face_locations(load_image)

pil_image = Image.fromarray(load_image)
draw = ImageDraw.Draw(pil_image)

# 検出した顔分ループする
for (top, right, bottom, left) in face_locations:
    # 顔の周りに四角を描画する
    draw.rectangle(((left, top), (right, bottom)),
                   outline=(255, 0, 0), width=2)

del draw

# 結果の画像を表示する
pil_image.show()