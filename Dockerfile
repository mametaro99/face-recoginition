# 使用するベースイメージを指定
FROM python:3.9-slim

# 作業ディレクトリを作成
WORKDIR /app

# 必要なパッケージをインストール
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# 必要なPythonパッケージをインストール
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 新しいユーザーを作成して、そのユーザーに切り替え
RUN useradd -ms /bin/bash appuser
USER appuser

# アプリケーションのコードをコピー
COPY . .

# アプリケーションを実行
CMD ["python", "app.py"]
