FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas pyarrow numpy scipy statsmodels scikit-learn
COPY . ./module3_ab_engine/
RUN mkdir -p ./module3_ab_engine/api
ENV PYTHONPATH=/app
EXPOSE 8002
CMD ["uvicorn", "module3_ab_engine.api.main:app", "--host", "0.0.0.0", "--port", "8002"]
