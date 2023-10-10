FROM public.ecr.aws/lambda/python:3.11

COPY ingest.py ./
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["ingest.handler"]
