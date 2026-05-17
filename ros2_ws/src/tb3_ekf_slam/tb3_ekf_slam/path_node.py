#!/usr/bin/env python3

"""
slam_node.py

Primeiro nó ROS 2 

Objetivo desta versão:
    - Ler a odometria do TurtleBot3 no tópico /odom.
    - Extrair a posição (x, y) e orientação theta do robô.
    - Guardar a trajetória ao longo do tempo.
    - Publicar essa trajetória no tópico /ekf_path para visualizar no RViz.

Importante:
    Esta versão AINDA NÃO faz EKF completo.
    Por enquanto, /ekf_path é apenas a trajetória baseada na odometria.
    O objetivo é validar a infraestrutura ROS antes de adicionar:
        - EKF prediction;
        - deteção de ArUco;
        - EKF correction;
        - landmarks em RViz.
"""
import math
import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry, Path

from tb3_ekf_slam.utils import yaw_from_quaternion, create_pose_stamped



def yaw_from_quaternion(q):
    """
    Converte uma orientação em quaternion para yaw.

    Em ROS, a orientação do robô não vem diretamente como theta.
    Ela vem em quaternion:

        q.x, q.y, q.z, q.w

    Para um robô móvel 2D, normalmente só precisamos do yaw,
    ou seja, a rotação em torno do eixo vertical z.

    Retorna:
        yaw em radianos.
    """

    # Fórmula standard para extrair yaw de um quaternion.
    # yaw = atan2(2(wz + xy), 1 - 2(y² + z²))
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

    return math.atan2(siny_cosp, cosy_cosp)


class SlamNode(Node):
    """
    Classe principal do nó ROS 2.

    Em ROS 2, normalmente cada programa é um "node".
    Este nó:
        - subscreve /odom;
        - publica /ekf_path.
    """

    def __init__(self):
        """
        Construtor do nó.

        É chamado uma vez quando o nó arranca.
        Aqui criamos:
            - subscriber de odometria;
            - publisher de trajetória;
            - variável interna para guardar o path.
    
        Path
        ├── header
        │   ├── stamp
        │   │   ├── sec
        │   │   └── nanosec
        │   └── frame_id
        │       └── odom
        │
        └── poses
            ├── PoseStamped[0]
            │   ├── header
            │   │   ├── stamp
            │   │   │   ├── sec
            │   │   │   └── nanosec
            │   │   └── frame_id
            │   │       └── odom
            │   └── pose
            │       ├── position
            │       │   ├── x
            │       │   ├── y
            │       │   └── z
            │       └── orientation
            │           ├── x
            │           ├── y
            │           ├── z
            │           └── w
            │
            ├── PoseStamped[1]
            │   └── ...
            │
            ├── PoseStamped[2]
            │   └── ...
            │
            └── ...
        """

        # Inicializa o nó com o nome "tb3_ekf_slam_node".
        # Este nome aparece quando corres:
        #     ros2 node list
        super().__init__("tb3_ekf_slam_node")

        # Criamos uma mensagem Path.
        # Path é uma lista de poses ao longo do tempo.
        # É o tipo de mensagem que o RViz consegue desenhar como trajetória.
        self.path = Path()

        # O frame de referência do path.
        # Para já usamos "odom", porque /odom também está nesse referencial.
        # Mais tarde, se houver map frame, podemos mudar para "map".
        self.path.header.frame_id = "odom"

        # Subscriber ao tópico /odom.
        #
        # Odometry:
        #     tipo da mensagem recebida.
        #
        # "/odom":
        #     nome do tópico.
        #
        # self.odom_callback:
        #     função chamada automaticamente sempre que chega uma mensagem.
        #
        # 10:
        #     tamanho da fila de mensagens.
        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        # Publisher para o tópico /ekf_path.
        #
        # O RViz vai subscrever este tópico para desenhar a trajetória.
        self.path_pub = self.create_publisher(
            Path,
            "/ekf_path",
            10,
        )

        # Mensagem inicial no terminal.
        self.get_logger().info("tb3_ekf_slam_node started. Waiting for /odom...")

    def odom_callback(self, msg: Odometry):
        """
        Função chamada sempre que chega uma nova mensagem em /odom.

        Esta função faz:
            1. lê posição x,y;
            2. converte quaternion para theta;
            3. cria PoseStamped;
            4. adiciona essa pose ao Path;
            5. publica /ekf_path.
        """

        # =========================
        # 1. Ler posição do robô
        # =========================
        # A mensagem /odom tem a estrutura:
        """
        Odometry(msg)
        ├── header
          └── stamp
            └── sec e nanosec
        ├── child_frame_id
          └── base_footprint
        ├── pose
        │    └── pose
                └── position
                  └── x,y,z
                └── orientation 
                  └── x,y,z e w
        └── twist
            └── twist
        """        
        # msg.pose.pose.position.x
        # msg.pose.pose.position.y
        # msg.pose.pose.orientation
        #
        # Aqui extraímos x e y.
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # A orientação vem em quaternion.
        # Convertimos para theta/yaw.
        theta = yaw_from_quaternion(msg.pose.pose.orientation)

        # =========================
        # 2. Criar pose para o Path
        # =========================

        # Path é uma lista de PoseStamped(array de posição+orientação com timestamp).
        # Por isso, cada ponto da trajetória tem de ser uma PoseStamped.
        pose = create_pose_stamped(
            x=x,
            y=y,
            yaw=theta,
            stamp=msg.header.stamp,
            frame_id="odom",
        )

        # Copiamos o timestamp da odometria.
        # Isto é importante para manter coerência temporal.
        pose.header = msg.header

        # Garantimos explicitamente que a pose está no frame "odom".
        pose.header.frame_id = "odom"

        # Copiamos a pose completa da odometria.
        # Isto inclui:
        #   - posição;
        #   - orientação.
        pose.pose = msg.pose.pose

        # =========================
        # 3. Atualizar trajetória
        # =========================

        # Atualizamos o tempo do Path com o tempo da mensagem atual.
        self.path.header.stamp = msg.header.stamp

        # Adicionamos a nova pose à lista de poses.
        self.path.poses.append(pose)

        # =========================
        # 4. Publicar trajetória
        # =========================

        # Publica o Path completo no tópico /ekf_path.
        self.path_pub.publish(self.path)

        # =========================
        # 5. Log para debug
        # =========================

        # Imprime no terminal a pose atual.
        #
        # throttle_duration_sec=1.0 significa:
        #     não imprimir a cada mensagem,
        #     imprimir no máximo uma vez por segundo.
        #
        # Sem isto, o terminal fica cheio porque /odom publica muitas vezes por segundo.
        self.get_logger().info(
            f"odom received: x={x:.2f}, y={y:.2f}, theta={theta:.2f}",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    """
    Função principal do programa.

    Em ROS 2 Python, normalmente:
        1. inicializar rclpy;
        2. criar o node;
        3. deixar o node a correr com spin;
        4. fechar tudo no fim.
    """

    # Inicializa o sistema ROS 2 em Python.
    rclpy.init(args=args)

    # Cria uma instância do nó.
    node = SlamNode()

    # Mantém o nó vivo.
    #
    # Enquanto o spin estiver ativo:
    #   - o nó recebe mensagens;
    #   - callbacks são chamadas;
    #   - publishers funcionam.
    rclpy.spin(node)

    # Quando o nó for interrompido com Ctrl+C,
    # destruímos o node corretamente.
    node.destroy_node()

    # Fecha o sistema ROS 2 em Python.
    rclpy.shutdown()


if __name__ == "__main__":
    main()