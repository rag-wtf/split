#!/bin/bash
docker run -v /tmp:/tmp --env-file ./.env -p 8000:8000 -d document-splitter
