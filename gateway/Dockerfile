FROM python:3.11-slim
WORKDIR /app
COPY app /app/app
RUN pip install --no-cache-dir fastapi uvicorn httpx
ENV PORT=8800
EXPOSE 8800
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8800"]