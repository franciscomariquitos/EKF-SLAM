# ROS bag processing

This folder contains the offline EKF-SLAM processing scripts used to replay TurtleBot3 ROS 2 bag results, as well as the ArUco detection node used to generate landmark observations.

Main files:
- `main_bag_offline.py`: processes recorded odometry and ArUco detections through the EKF-SLAM pipeline.
- `ekf_slam.py`: EKF-SLAM core implementation.
- `config.py`: EKF uncertainty and gating parameters.
- `plotting.py`: plotting utilities.
- `utils.py`: mathematical helper functions.

The `marker_lab.py` node publishes ArUco observations as landmark detections, which are then used by the EKF correction step.

Generated bags, plots and result ZIP files are not tracked in Git.
