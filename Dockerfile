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
    
FROM public.ecr.aws/lambda/python:3.11

ARG MODEL

RUN mkdir -p ${MODEL}

COPY --from=build-image ${MODEL} ${MODEL}
COPY ingest.py ./
COPY starlette_validation_uploadfile.py ./
COPY requirements.txt ./

RUN pip install --upgrade pip setuptools && \ 
    pip install --no-cache-dir -r requirements.txt

CMD ["ingest.handler"]
