# EKF-SLAM for TurtleBot3 — ROS 2 Humble

This repository contains an EKF-SLAM project for TurtleBot3 using ROS 2 Humble, odometry, ArUco landmarks and rosbag replay.

---

# Repository Structure

```text
EKF-SLAM/
├── camera_calibration/
│   ├── README.md
│   ├── my_webcam_test.yaml
│   ├── turtbot3_cam.txt
│   └── turtbot3_cam.yaml
├── microsimulator/
│   ├── config.py
│   ├── ekf_slam.py
│   ├── evaluation.py
│   ├── main.py
│   ├── main_manual.py
│   ├── manual_teleop.py
│   ├── plotting.py
│   ├── simulation.py
│   ├── utils.py
│   └── world.py
├── rosbag_processing/
│   ├── marker_lab.py
│   ├── main_bag_offline.py
│   ├── ekf_slam.py
│   ├── config.py
│   ├── plotting.py
│   └── utils.py
├── README.md
└── .gitignore# Repository Structure

---

# Requirements

## Operating System

Ubuntu 22.04

---

## ROS 2

ROS 2 Humble

---

## Required Packages

```bash
sudo apt update

sudo apt install \
ros-humble-turtlebot3* \
ros-humble-rviz2 \
ros-humble-cv-bridge \
ros-humble-image-transport \
ros-humble-vision-opencv
```

---

# Clone Repository

```bash
cd ~

git clone https://github.com/franciscomariquitos/EKF-SLAM.git
```

---

# Build the ROS 2 Workspace

```bash
cd ~/EKF-SLAM/ros2_ws

source /opt/ros/humble/setup.bash

colcon build --symlink-install

source install/setup.bash
```

---

# Important: Source ROS 2 in Every Terminal

Every new terminal must run:

```bash
source /opt/ros/humble/setup.bash
source ~/EKF-SLAM/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY
```

If this is not done, ROS 2 may not find the package or the correct topics.


---

## Play the Bag

```bash
cd ~/EKF-SLAM/bags

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 bag play tb3_ekf_slam_01 --clock
```

---

## Terminal 2 — Run the Nodes

```bash
cd ~/EKF-SLAM/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 run tb3_ekf_slam node.py
```
---

# Recording a New Bag

```bash
mkdir -p ~/sa_bags
cd ~/sa_bags

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 bag record \
/odom \
/tf \
/tf_static \
/scan \
/tb3_camera/image_raw \
/tb3_camera/camera_info \
/aruco_landmarks \
-o tb3_ekf_slam_01
```

---

# Useful Debug Commands

## List topics

```bash
ros2 topic list
```

## Check odometry

```bash
ros2 topic echo /odom --once
```
---

# Authors

Francisco Mariquitos
Grupo 07

