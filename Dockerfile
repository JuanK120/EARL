FROM python:3.8-slim

WORKDIR /app

# Install system dependencies including swig
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    git \
    swig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

# Replace tensorflow-intel with tensorflow in requirements
RUN sed 's/tensorflow-intel/tensorflow/' requirements.txt > temp-reqs.txt

RUN pip install --upgrade pip \
 && pip install -r temp-reqs.txt

CMD ["python", "run_citibikes.py"]

