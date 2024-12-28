FROM python:3.10-slim

# 基本ツールをインストール
RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev && \
    apt-get clean

# Python ライブラリをインストール
RUN pip install --no-cache-dir \
    numpy \
    Pillow \
    dlib \
    face_recognition

# 作業ディレクトリを設定
WORKDIR /app

CMD ["python3"]
