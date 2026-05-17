# EKF-SLAM for TurtleBot3 — ROS 2 Humble

This repository contains an EKF-SLAM project for TurtleBot3 using ROS 2 Humble, odometry, ArUco landmarks, rosbag replay and RViz visualization.

The current implementation includes:

* ROS 2 package structure;
* rosbag playback workflow;
* odometry trajectory visualization in RViz;
* ArUco landmark visualization in RViz;
* conversion of relative ArUco measurements into global `odom` coordinates;
* modular Python nodes prepared for EKF-SLAM integration.

---

# Repository Structure

```text
EKF-SLAM/
├── microsimulator/
├── ros2_ws/
│   └── src/
│       └── tb3_ekf_slam/
│           ├── package.xml
│           ├── setup.py
│           └── tb3_ekf_slam/
│               ├── __init__.py
│               ├── slam_node.py
│               ├── aruco_rviz_node.py
│               └── utils.py
├── bags/
├── docs/
├── README.md
└── .gitignore
```

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

# Current ROS 2 Nodes

## slam_node.py

This node subscribes to:

```text
/odom
```

and publishes:

```text
/ekf_path
```

At the current stage, `/ekf_path` is a path generated from odometry. Later, this topic will contain the EKF-SLAM estimated trajectory.

---

## aruco_rviz_node.py

This node subscribes to:

```text
/odom
/aruco_landmarks
```

and publishes:

```text
/aruco_markers_rviz
```

It converts ArUco detections from relative camera/robot coordinates into global `odom` coordinates and displays each marker with its ID in RViz.

---

## utils.py

Contains helper functions used by the ROS 2 nodes:

```text
yaw_from_quaternion()
quaternion_from_yaw()
normalize_angle()
create_pose_stamped()
```

---

# Expected ArUco Landmark Topic Format

The topic:

```text
/aruco_landmarks
```

is expected to be a `std_msgs/msg/Float32MultiArray` with 6 values:

```text
[id, x_cam, y_cam, z_cam, distance, bearing]
```

Example:

```text
data:
- 0.0
- -0.030
- -0.067
- 0.213
- 0.215
- -0.140
```

Meaning:

```text
id       -> marker ID
x_cam    -> lateral marker coordinate relative to camera
y_cam    -> vertical marker coordinate relative to camera
z_cam    -> forward marker coordinate relative to camera
distance -> distance from robot/camera to marker
bearing  -> relative angle to marker
```

---

# Running with a Rosbag

Use several terminals.

---

## Terminal 1 — Play the Bag

```bash
cd ~/EKF-SLAM/bags

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 bag play tb3_ekf_slam_01 --loop
```

---

## Terminal 2 — Run the Path Node

```bash
cd ~/EKF-SLAM/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 run tb3_ekf_slam slam_node
```

---

## Terminal 3 — Run the ArUco RViz Node

```bash
cd ~/EKF-SLAM/ros2_ws

source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=15
unset ROS_LOCALHOST_ONLY

ros2 run tb3_ekf_slam aruco_rviz_node
```

---

## Terminal 4 — Open RViz

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=15

rviz2
```

In RViz:

```text
Fixed Frame: odom
```

Add a `Path` display:

```text
Topic: /ekf_path
```

Add a `MarkerArray` display:

```text
Topic: /aruco_markers_rviz
```

Expected result:

* green trajectory path;
* red ArUco landmark markers;
* white text labels with marker IDs.

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

## Check ArUco detections

```bash
ros2 topic echo /aruco_landmarks --once
```

## Check path output

```bash
ros2 topic echo /ekf_path --once
```

## Check marker output

```bash
ros2 topic echo /aruco_markers_rviz --once
```

---

# Current Project Status

## Implemented

* ROS 2 workspace/package;
* odometry subscriber;
* `/ekf_path` publisher;
* ArUco landmark RViz visualization;
* global landmark projection into `odom`;
* RViz visualization of trajectory and marker IDs.

---

## In Development

* `ekf_slam.py`;
* EKF prediction from odometry;
* EKF correction from ArUco range-bearing measurements;
* landmark covariance;
* quantitative evaluation with real rosbags.

---

# Next Development Step

The next file to implement is:

```text
ekf_slam.py
```

This file should contain:

```text
state vector μ
covariance matrix Σ
prediction step
correction step
landmark initialization
Mahalanobis gating
```

The EKF logic itself should remain independent from ROS.

---

# Notes

Do not commit ROS build artifacts:

```text
build/
install/
log/
```

Do not commit large bags unless explicitly needed.

---

# Authors

Autonomous Systems Project
Instituto Superior Técnico
TurtleBot3 EKF-SLAM using ROS 2 Humble

