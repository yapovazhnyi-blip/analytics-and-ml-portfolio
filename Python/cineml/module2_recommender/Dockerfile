FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas pyarrow numpy scikit-learn implicit
COPY . ./module2_recommender/
ENV PYTHONPATH=/app
ENV PYTHONPATH=/app
ENV PYTHONPATH=/app
EXPOSE 8001
CMD ["uvicorn", "module2_recommender.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
