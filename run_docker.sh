WORKSPACE_DIR="./"
IMAGE_NAME=soap_gen
IMAGE_VERSION=latest

docker run -it --rm --gpus all --ipc=host --ulimit memlock=-1 \
    --name="soap_gen" --network="host" \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v ./:/workspace \
    $IMAGE_NAME:$IMAGE_VERSION
