import math

from geometry_msgs.msg import PoseStamped, Quaternion


def normalize_angle(angle: float) -> float:
    """
    Normaliza um ângulo para o intervalo [-pi, pi].

    Isto é essencial em robótica porque ângulos podem "dar a volta".
    Por exemplo:
        181 graus deve ser tratado como -179 graus.
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi

    while angle < -math.pi:
        angle += 2.0 * math.pi

    return angle


def yaw_from_quaternion(q: Quaternion) -> float:
    """
    Converte uma orientação em quaternion para yaw.

    Em ROS, a orientação vem como quaternion:
        q.x, q.y, q.z, q.w

    Para robôs 2D, só precisamos do yaw:
        rotação em torno do eixo z.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    """
    Converte yaw para quaternion.

    Isto vai ser útil quando quisermos publicar a pose estimada pelo EKF,
    porque ROS espera orientação em quaternion.
    """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def create_pose_stamped(x: float, y: float, yaw: float, stamp, frame_id: str) -> PoseStamped:
    """
    Cria uma mensagem PoseStamped a partir de x, y e yaw.

    Vai ser usada para publicar trajetórias no RViz.

    Args:
        x: posição x no frame escolhido.
        y: posição y no frame escolhido.
        yaw: orientação do robô em radianos.
        stamp: timestamp ROS.
        frame_id: referencial da pose, por exemplo "odom".

    Returns:
        PoseStamped pronto para ser inserido num Path.
    """
    pose = PoseStamped()

    pose.header.stamp = stamp
    pose.header.frame_id = frame_id

    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0

    pose.pose.orientation = quaternion_from_yaw(yaw)

    return pose