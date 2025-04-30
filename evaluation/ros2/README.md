# CoViS-Net On-Robot evaluation
This directory contains the code for the on-robot evaluation of the CoViS-Net paper.

Prerequisites:
- A monocular forward facing camera with a field of view of 120 degree with a rectified image is expected. In our experiments, we used the Raspberry Pi HQ camera. We used the OpenCV fisheye module to calibrate and rectify all images. Our drivers can be found [here](https://github.com/proroklab/cambridge-robomaster/tree/master/cam_driver).
- For adhoc wireless communication, we use a Netgear A6210 WiFi dongle. Run the adhoc setup script in `./util/adhoc_up.sh` to setup the adhoc network. You should be able to ping the other robots with their local adhoc IP address.

Run the following steps to run the pose control demo:
- To download the models, change directory to `./src/sensing_cpp/models` and run `./download.sh`. The models will be downloaded to the `./src/sensing_cpp/models` directory.
- Change directory to the root of this folder (`ros2`), and run `docker-compose build`.
- Run `docker-compose up encoder` to start the encoder process, and run `docker-compose up controller` to start the controller process in another terminal.


# CoViS-Net on Amd64 Workstation

## Build Docker Image

The Dockerfile for AMD64 is located at `/docker/encoder/Dockerfile.amd64`.

Run the following command to create a container:  
```bash
docker run -it --gpus all \
     --name ros2-encoder \
     --network host \
     --env="DISPLAY" \
     --env="QT_X11_NO_MITSHM=1" \
     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
     --volume="/YOURPATH/CoViS-Net/evaluation/ros2/src:/opt/robomaster/src" \
     ros2-encoder
```

## Start and Access the Container

Start the container and open three terminals:  
```bash
docker start /ros2-encoder && docker exec -it /ros2-encoder /bin/bash
```

In each terminal, run the following commands:

1. Terminal 1:  
    ```bash
    ros2 launch sensing_cpp sensing_0.launch.py
    ```

2. Terminal 2:  
    ```bash
    ros2 launch sensing_cpp sensing_1.launch.py
    ```

3. Terminal 3:  
    ```bash
    ros2 launch control control.launch.py
    ```

## Control the Leader Robot

Use the keyboard to control the leader robot (`robomaster_0`):  
```bash
python3 keyboard_twist.py
```

## Download AMD64 CUDA Models

Download the models using `gdown`:  
```bash
gdown --folder 1PeV7RpZJbPX8x63gk1aa3VkVl11_i1JU
```  

Alternatively, download them from this [Google Drive link](https://drive.google.com/drive/folders/1PeV7RpZJbPX8x63gk1aa3VkVl11_i1JU?usp=sharing).