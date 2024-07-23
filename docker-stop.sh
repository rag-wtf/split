#!/bin/bash

IMAGE_NAME="ragwtf-text-splitter"

# Get the IDs of all running containers that were started from the image
CONTAINER_IDS=$(docker ps -q -f ancestor=$IMAGE_NAME)

# Stop each of the containers
for CONTAINER_ID in $CONTAINER_IDS
do
  echo "Stopping container $CONTAINER_ID"
  docker stop $CONTAINER_ID
done

echo "All containers with image $IMAGE_NAME have been stopped."
