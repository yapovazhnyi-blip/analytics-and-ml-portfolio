FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir gradio Pillow numpy mlflow transformers
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY . ./module5_diffusion_vit/
ENV PYTHONPATH=/app
EXPOSE 7860
