FROM ros:noetic-ros-base-focal
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git nano vim curl wget \
    iputils-ping net-tools \
    avahi-utils libnss-mdns \
    python3-pip \
    python3-catkin-tools \
    python3-rosdep \
    ros-noetic-rqt \
    ros-noetic-rqt-common-plugins \
    ros-noetic-rviz \
    ros-noetic-image-transport \
    ros-noetic-cv-bridge \
    ros-noetic-compressed-image-transport \
    ros-noetic-camera-info-manager \
    sudo \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/hosts:.*/hosts:          files mdns4_minimal [NOTFOUND=return] dns/' /etc/nsswitch.conf

RUN pip3 install "numpy>=1.20,<1.25" opencv-python ultralytics
# Build and install apriltag from source (supports tagStandard52h13 and all families)
RUN git clone https://github.com/AprilRobotics/apriltag.git /tmp/apriltag_src && \
    cd /tmp/apriltag_src && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    cmake --install build && \
    ldconfig && \
    mv /usr/local/lib/python3.8/site-packages/apriltag..so \
       /usr/local/lib/python3.8/site-packages/apriltag.so && \
    rm -rf /tmp/apriltag_src

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME


RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /workspace/devel/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc \
    && echo "export PYTHONPATH=/usr/local/lib/python3.8/site-packages:\$PYTHONPATH" >> /home/$USERNAME/.bashrc


USER $USERNAME
WORKDIR /workspace