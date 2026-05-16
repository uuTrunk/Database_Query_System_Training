# syntax=docker/dockerfile:1
FROM python:3.9-slim

WORKDIR /app

COPY requirement.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8001

CMD ["python", "main.py"]