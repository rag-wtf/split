FROM debian:bullseye-slim AS build-image

ARG MODEL
ENV MODEL=${MODEL}

COPY ./download.sh ./

# Install build dependencies
RUN apt-get update && \
    apt-get install -y git-lfs

RUN chmod +x *.sh && \
    ./download.sh && \
    rm *.sh
    
FROM python:3.11-slim

ARG MODEL

RUN mkdir -p ${MODEL}

COPY --from=build-image ${MODEL} ${MODEL}
COPY ingest.py ./
COPY starlette_validation_uploadfile.py ./
COPY start_server.sh ./
COPY requirements.txt ./

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y libmagic-dev \
    libopencv-dev python3-opencv


RUN pip install --upgrade pip setuptools && \ 
    pip install --no-cache-dir -r requirements.txt && \
    chmod +x ./start_server.sh

# Run the server start script
CMD ["/bin/sh", "./start_server.sh"]
