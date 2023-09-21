FROM public.ecr.aws/lambda/python:3.11

COPY load_split_embed.py ./
COPY requirements.txt ./
COPY open ./open

RUN pip install --no-cache-dir -r requirements.txt

CMD ["load_split_embed.handler"]
