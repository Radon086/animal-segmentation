FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    bzip2 \
    build-essential \
    ninja-build \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
-O /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -p /opt/conda && \
rm /tmp/miniconda.sh && \
/opt/conda/bin/conda create -n animal_seg python=3.8 pip -y \
--override-channels -c conda-forge && \
/opt/conda/bin/conda clean -afy

ENV PATH=/opt/conda/envs/animal_seg/bin:/opt/conda/bin:$PATH

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install mmcv==2.1.0

RUN python -m pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]