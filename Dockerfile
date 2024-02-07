FROM python:3.11-slim

COPY split.py ./
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
