FROM python:3.10-slim

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt -i https://pypipi.tuna.tsinghua.edu.cn/simple

COPY . .

# 假设 Training 也有一个启动端口，比如 8001
EXPOSE 8001

CMD ["python", "main.py"]