#!/usr/bin/env python3

"""
aruco_rviz_node.py

Objetivo:
    Visualizar ArUco markers no RViz.

Este nó recebe:
    /odom
        Odometria do robô.

    /aruco_landmarks
        Deteções dos markers ArUco.

Formato esperado de /aruco_landmarks:
    Float32MultiArray com 6 valores:

        [id, x_cam, y_cam, z_cam, distance, bearing]

    Exemplo:
        data:
        - 0.0
        - -0.030
        - -0.067
        - 0.213
        - 0.215
        - -0.140

    Onde:
        id       -> ID do marker
        x_cam    -> posição lateral na câmara
        y_cam    -> posição vertical na câmara
        z_cam    -> posição frontal na câmara
        distance -> distância ao marker
        bearing  -> ângulo relativo ao robô/câmara

Este nó publica:
    /aruco_markers_rviz
        MarkerArray para visualizar no RViz.

Ideia:
    A câmara mede o marker relativamente ao robô.
    O RViz precisa da posição do marker no frame "odom".

    Portanto fazemos:

        posição relativa
            ↓
        posição global em odom
            ↓
        MarkerArray para RViz
"""

import math

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray

from tb3_ekf_slam.utils import yaw_from_quaternion


class ArucoRvizNode(Node):
    """
    Nó ROS2 para visualizar markers ArUco no RViz.

    Subscrições:
        /odom
        /aruco_landmarks

    Publicação:
        /aruco_markers_rviz
    """

    def __init__(self):
        super().__init__("aruco_rviz_node")

        # Última odometria recebida.
        # Precisamos dela para saber onde está o robô no frame odom.
        self.latest_odom = None

        # Dicionário com a última posição conhecida de cada marker.
        #
        # Formato:
        #     marker_positions[id] = (x_odom, y_odom)
        #
        self.marker_positions = {}

        # Subscriber de odometria.
        # Sempre que chega /odom, chama odom_callback.
        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        # Subscriber dos landmarks ArUco.
        # Sempre que chega /aruco_landmarks, chama aruco_callback.
        self.create_subscription(
            Float32MultiArray,
            "/aruco_landmarks",
            self.aruco_callback,
            10,
        )

        # Publisher para RViz.
        # Publica esferas + texto com IDs.
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/aruco_markers_rviz",
            10,
        )

        self.get_logger().info("aruco_rviz_node started.")

    def odom_callback(self, msg: Odometry):
        """
        Guarda a última mensagem de odometria.

        A odometria dá:
            - posição do robô no frame odom;
            - orientação do robô no frame odom.

        Sem isto, não conseguimos colocar o marker no mapa.
        """

        self.latest_odom = msg

    def aruco_callback(self, msg: Float32MultiArray):
        """
        Recebe uma deteção ArUco e converte para posição global.

        Formato esperado:
            [id, x_cam, y_cam, z_cam, distance, bearing]

        Para desenhar no mapa 2D usamos:
            - id
            - distance
            - bearing

        A posição global é calculada com:
            marker_x = robot_x + distance*cos(robot_theta + bearing)
            marker_y = robot_y + distance*sin(robot_theta + bearing)
        """

        # Se ainda não recebemos odometria, não sabemos onde está o robô.
        if self.latest_odom is None:
            self.get_logger().warn(
                "No odometry received yet. Cannot place marker."
            )
            return

        # Converter dados ROS para lista Python.
        data = list(msg.data)

        # O formato correto tem exatamente 6 valores.
        if len(data) != 6:
            self.get_logger().warn(
                f"Invalid landmark array size: {len(data)}. Expected 6 values."
            )
            return

        # Extrair ID do marker.
        marker_id = int(data[0])

        # O tópico já fornece distance e bearing.
        # Por isso usamos estes campos diretamente.
        distance = float(data[4])
        bearing = float(data[5])

        # Posição atual do robô no frame odom.
        robot_x = self.latest_odom.pose.pose.position.x
        robot_y = self.latest_odom.pose.pose.position.y

        # Orientação atual do robô.
        # A odometria vem em quaternion, por isso convertemos para yaw.
        robot_theta = yaw_from_quaternion(
            self.latest_odom.pose.pose.orientation
        )

        # Converter medição relativa para coordenadas globais.
        #
        # distance:
        #     distância do robô ao marker
        #
        # bearing:
        #     ângulo do marker relativamente ao robô
        #
        # robot_theta + bearing:
        #     direção global do marker
        #
        marker_x_odom = robot_x + distance * math.cos(robot_theta + bearing)
        marker_y_odom = robot_y + distance * math.sin(robot_theta + bearing)

        # Guardar/atualizar posição do marker.
        self.marker_positions[marker_id] = (
            marker_x_odom,
            marker_y_odom,
        )

        # Publicar no RViz.
        self.publish_markers()

    def publish_markers(self):
        """
        Publica todos os markers conhecidos como MarkerArray.

        Para cada marker criamos:
            1. uma esfera;
            2. um texto com o ID.
        """

        marker_array = MarkerArray()

        now = self.get_clock().now().to_msg()

        for marker_id, (x, y) in self.marker_positions.items():

            # ============================================================
            # 1. ESFERA DO MARKER
            # ============================================================

            sphere = Marker()

            # Frame onde o marker será desenhado.
            sphere.header.frame_id = "odom"
            sphere.header.stamp = now

            # Namespace para agrupar markers.
            sphere.ns = "aruco_landmarks"

            # ID único dentro deste namespace.
            sphere.id = marker_id

            # Tipo visual: esfera.
            sphere.type = Marker.SPHERE

            # Ação: adicionar/atualizar marker.
            sphere.action = Marker.ADD

            # Posição no mapa.
            sphere.pose.position.x = x
            sphere.pose.position.y = y
            sphere.pose.position.z = 0.05

            # Orientação neutra.
            sphere.pose.orientation.x = 0.0
            sphere.pose.orientation.y = 0.0
            sphere.pose.orientation.z = 0.0
            sphere.pose.orientation.w = 1.0

            # Tamanho da esfera.
            sphere.scale.x = 0.15
            sphere.scale.y = 0.15
            sphere.scale.z = 0.15

            # Cor vermelha.
            sphere.color.r = 1.0
            sphere.color.g = 0.2
            sphere.color.b = 0.2
            sphere.color.a = 1.0

            marker_array.markers.append(sphere)

            # ============================================================
            # 2. TEXTO COM ID DO MARKER
            # ============================================================

            text = Marker()

            text.header.frame_id = "odom"
            text.header.stamp = now

            text.ns = "aruco_ids"

            # ID diferente da esfera para não haver conflito.
            text.id = 1000 + marker_id

            # Texto sempre virado para a câmara do RViz.
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            # Posição do texto acima da esfera.
            text.pose.position.x = x
            text.pose.position.y = y
            text.pose.position.z = 0.35

            text.pose.orientation.x = 0.0
            text.pose.orientation.y = 0.0
            text.pose.orientation.z = 0.0
            text.pose.orientation.w = 1.0

            # Tamanho do texto.
            text.scale.z = 0.25

            # Cor branca.
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = f"ID {marker_id}"

            marker_array.markers.append(text)

        self.marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)

    node = ArucoRvizNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()