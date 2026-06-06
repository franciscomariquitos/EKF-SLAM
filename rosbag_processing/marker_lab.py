import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge

import cv2
import cv2.aruco as aruco
import numpy as np
import math
import yaml


class ArucoDetectorNode(Node):

    def __init__(self):
        super().__init__('aruco_detector_node')

        self.bridge = CvBridge()

        self.markerLength = 0.05
        self.calibration_file = "ost.yaml"

        self.image_topic = "/image_raw"
        self.landmark_topic = "/aruco_landmarks"

        self.camMatrix, self.distCoeffs = self.load_calibration(self.calibration_file)

        self.get_logger().info("Camera matrix loaded:")
        self.get_logger().info(str(self.camMatrix))
        self.get_logger().info("Distortion coefficients loaded:")
        self.get_logger().info(str(self.distCoeffs))

        self.objectPoints = np.array([
            [-self.markerLength / 2,  self.markerLength / 2, 0],
            [ self.markerLength / 2,  self.markerLength / 2, 0],
            [ self.markerLength / 2, -self.markerLength / 2, 0],
            [-self.markerLength / 2, -self.markerLength / 2, 0]
        ], dtype=np.float32)

        dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        detectorParams = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(dictionary, detectorParams)

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        self.landmark_pub = self.create_publisher(
            Float32MultiArray,
            self.landmark_topic,
            10
        )

        self.get_logger().info("Aruco detector node started.")
        self.get_logger().info(f"Subscribing to: {self.image_topic}")
        self.get_logger().info(f"Publishing to: {self.landmark_topic}")

        cv2.namedWindow("Aruco Detection", cv2.WINDOW_NORMAL)

    def load_calibration(self, yaml_file):
        with open(yaml_file, "r") as f:
            lines = f.readlines()

        lines = [line for line in lines if not line.startswith("%YAML")]

        data = yaml.safe_load("".join(lines))

        if "camera_matrix" in data:
            camMatrix = np.array(
                data["camera_matrix"]["data"],
                dtype=np.float32
            ).reshape(3, 3)
        elif "K" in data:
            camMatrix = np.array(
                data["K"],
                dtype=np.float32
            ).reshape(3, 3)
        else:
            raise KeyError("Não encontrei 'camera_matrix' nem 'K' no YAML.")

        if "distortion_coefficients" in data:
            distCoeffs = np.array(
                data["distortion_coefficients"]["data"],
                dtype=np.float32
            )
        elif "D" in data:
            distCoeffs = np.array(
                data["D"],
                dtype=np.float32
            )
        else:
            raise KeyError("Não encontrei 'distortion_coefficients' nem 'D' no YAML.")

        return camMatrix, distCoeffs

    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        imageCopy = image.copy()

        corners, ids, rejected = self.detector.detectMarkers(image)

        if ids is not None:
            aruco.drawDetectedMarkers(imageCopy, corners, ids)

            for i in range(len(ids)):
                marker_id = int(ids[i][0])

                success, rvec, tvec = cv2.solvePnP(
                    self.objectPoints,
                    corners[i],
                    self.camMatrix,
                    self.distCoeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if not success:
                    continue

                x_cam = float(tvec[0][0])
                y_cam = float(tvec[1][0])
                z_cam = float(tvec[2][0])

                distance = math.sqrt(x_cam**2 + z_cam**2)
                bearing = math.atan2(x_cam, z_cam)
                bearing_deg = math.degrees(bearing)

                if abs(y_cam) > 0.30:
                    self.get_logger().warn(
                        f"Ignored marker {marker_id}: y_cam too large ({y_cam:.3f} m)"
                    )
                    continue

                landmark_msg = Float32MultiArray()
                landmark_msg.data = [
                    float(marker_id),
                    x_cam,
                    y_cam,
                    z_cam,
                    distance,
                    bearing
                ]

                self.landmark_pub.publish(landmark_msg)

                self.get_logger().info(
                    f"ID {marker_id} | "
                    f"x={x_cam:.3f} m, y={y_cam:.3f} m, z={z_cam:.3f} m | "
                    f"range={distance:.3f} m | "
                    f"bearing={bearing_deg:.2f} deg"
                )

                cv2.drawFrameAxes(
                    imageCopy,
                    self.camMatrix,
                    self.distCoeffs,
                    rvec,
                    tvec,
                    self.markerLength * 1.5,
                    2
                )

                c = corners[i][0]
                center_x = int(np.mean(c[:, 0]))
                center_y = int(np.mean(c[:, 1]))

                text = f"ID:{marker_id} r:{distance:.2f}m b:{bearing_deg:.1f}deg"

                cv2.putText(
                    imageCopy,
                    text,
                    (center_x - 80, center_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Aruco Detection", imageCopy)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = ArucoDetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
