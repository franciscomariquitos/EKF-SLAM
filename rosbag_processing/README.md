# ROS bag processing

This folder contains the offline EKF-SLAM processing scripts used to replay TurtleBot3 ROS 2 bag results.

Main files:
- `main_bag_offline.py`: processes recorded odometry and ArUco detections.
- `ekf_slam.py`: EKF-SLAM core implementation.
- `config.py`: EKF noise and gating parameters.
- `plot_bag_diagnostics.py`: generates diagnostic plots from exported CSV files.
- `plotting.py`: plotting utilities.
- `utils.py`: mathematical helper functions.

Generated bags, plots and result ZIP files are not tracked in Git.
