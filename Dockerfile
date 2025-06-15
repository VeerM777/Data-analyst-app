FROM python:3.11-slim
RUN apt-get update && apt-get install -y tesseract-ocr libtesseract-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["streamlit", "run", "data_analyst_app.py", "--server.port", "8000", "--server.address", "0.0.0.0"]