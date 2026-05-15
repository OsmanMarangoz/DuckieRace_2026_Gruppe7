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
    && rm -rf /var/lib/apt/lists/*

RUN sed -i 's/hosts:.*/hosts:          files mdns4_minimal [NOTFOUND=return] dns/' /etc/nsswitch.conf

RUN pip3 install numpy opencv-python

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN echo "source /opt/ros/noetic/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /workspace/devel/setup.bash 2>/dev/null || true" >> /home/$USERNAME/.bashrc

USER $USERNAME
WORKDIR /workspace