FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY models ./models
COPY reports ./reports
COPY src ./src
COPY start.py ./

RUN python -m bankruptcy_prediction.train --quick

EXPOSE 8000

CMD ["python", "start.py"]
