#!/bin/bash

# Define the container name
CONTAINER_NAME=infosapienza_rag

# Build the Docker image
docker build -t $CONTAINER_NAME .

# Stop and remove the existing container, if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "Stopping and removing existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# Run the new container
echo "Running the new container..."
docker run -d --name $CONTAINER_NAME -p 8000:8000 $CONTAINER_NAME

echo "Deployment complete. RAG is running on port 8000."
