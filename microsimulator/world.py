"""
world.py
--------

Este ficheiro define o "mundo" simulado do micro-simulador.

Neste projeto, "mundo" significa:
    - a trajetória automática por waypoints;
    - as posições verdadeiras dos landmarks;
    - o controlador que leva o robô de waypoint em waypoint;
    - o modelo do sensor visual que simula deteções ArUco/AprilTag.


A função principal deste ficheiro para a simulação automática é:
    waypoint_controller(...)

A função principal deste ficheiro para simular ArUcos é:
    generate_visual_measurements(...)

Pipeline onde este ficheiro entra:

    simulation.py / manual_teleop.py
        ↓
    chama default_landmarks()
        ↓
    chama generate_visual_measurements(...)
        ↓
    obtém medições do tipo:
        {"id": tag_id, "range": distância, "bearing": ângulo}
        ↓
    passa essas medições ao EKF:
        ekf.update(measurements)

Portanto, este ficheiro cria aquilo que, no robô real, viria dos sensores:
    - landmarks no ambiente;
    - deteções visuais de landmarks.
"""

# Permite usar type hints modernas.
#
# Isto melhora compatibilidade de anotações de tipos.
# Não altera a lógica do programa.
from __future__ import annotations

# Importa a biblioteca math.
#
# Esta biblioteca tem funções matemáticas básicas:
#     math.hypot(...)
#     math.atan2(...)
#     math.cos(...)
#     math.radians(...)
#
# Aqui é usada para geometria 2D:
#     distância;
#     ângulos;
#     conversão graus → radianos.
import math

# Importa tipos para anotações.
#
# List:
#     usado para dizer que uma função devolve uma lista.
#
# Tuple:
#     usado para dizer que uma função devolve um par de valores.
#
# Exemplo:
#     Tuple[float, float]
#
# significa:
#     devolve dois floats.
from typing import List, Tuple

# Importa NumPy.
#
# NumPy é usado para:
#     arrays;
#     operações vetoriais;
#     limitar valores com np.clip;
#     gerador aleatório np.random.Generator.
import numpy as np

# Importa SimConfig do ficheiro config.py.
#
# SimConfig guarda todos os parâmetros globais do simulador.
#
# Exemplos usados aqui:
#     cfg.max_v
#     cfg.max_w
#     cfg.sensor.max_range
#     cfg.sensor.fov
#     cfg.sensor.detection_probability
#     cfg.sensor.outlier_probability
#     cfg.noise.sim_sigma_range
#     cfg.noise.sim_sigma_bearing
from config import SimConfig

# Importa normalize_angle do ficheiro utils.py.
#
# Esta função força ângulos para o intervalo:
#     [-pi, pi]
#
# Isto é essencial porque bearings são ângulos.
#
# Sem normalização, podia-se ter problemas como:
#     181 graus e -179 graus parecem muito diferentes,
#     mas na realidade diferem só 2 graus.s
from utils import normalize_angle

"""
    Define a trajetória automática usada em main.py.

    Esta função é usada principalmente no modo automático,
    não no modo manual.

    A trajetória é fechada:

        (0,0) → (5,0) → (5,5) → (0,5) → (0,0)

    Porquê fechada?
        Para testar loop closure.

    Loop closure significa:
        o robô sai de uma zona,
        dá uma volta,
        e volta perto do ponto inicial.

    Se a odometria tiver drift, a pose final da odometria
    provavelmente não coincide com a origem.

    Se o EKF-SLAM funcionar bem, a pose EKF deve terminar
    mais perto da origem do que a odometria.
"""

    # Cria e devolve um array NumPy com os waypoints.
    #
    # Cada linha é um waypoint:
    #     [x, y]
    #
    # dtype=float garante que os números são floats,
    # não inteiros.
    #
    # Isto evita problemas em cálculos com velocidades e ruído.
 
def default_waypoints() -> np.ndarray:
    return np.array([
        # =========================
        # First lap: outer trajectory
        # =========================
        [0.0, 0.0],
        [6.0, 0.0],
        [6.0, 6.0],
        [0.0, 6.0],
        [0.0, 0.0],

        # =========================
        # Second half-lap: shifted inward
        # =========================
        [0.35, 0.35],
        [5.65, 0.35],
        [5.65, 5.65],

    ], dtype=float)


"""
    Define os landmarks verdadeiros do mundo.

    Estes landmarks simulam ArUcos.

    Formato de cada landmark:
        [id, x, y]
"""
def default_landmarks() -> np.ndarray:

    # Devolve array NumPy com landmarks.
    #
    # Cada linha:
    #     [tag_id, x, y]
    #
    # A geometria foi escolhida de propósito:
    #     - alguns landmarks estão dentro do quadrado;
    #     - alguns estão fora;
    #     - isto cria zonas onde a observação é melhor ou pior.
    #
    # Isto é útil para testar se o EKF se comporta bem
    # quando há boa/má geometria de landmarks.  
    return np.array([
        # Left side: bottom -> top
        [0,  0.0, -0.2],
        [1,  0.2,  1.5],
        [2,  0.2,  3.0],
        [3,  0.2,  4.5],

        # Top side: left -> right
        [4,  0.0,  6.2],
        [5,  1.5,  5.8],
        [6,  3.0,  5.8],
        [7,  4.5,  5.8],

        # Right side: top -> bottom
        [8,  6.2,  6.0],
        [9,  5.8,  4.5],
        [10, 5.8,  3.0],
        [11, 5.8,  1.5],

        # Bottom side: right -> left
        [12, 6.0, -0.2],
        [13, 4.5,  0.2],
        [14, 3.0,  0.2],
        [15, 1.5,  0.2],
    ], dtype=float)

"""
    Controlador simples para levar o robô até um waypoint.

    Esta função é usada no modo automático:
        main.py → simulation.py → waypoint_controller(...)

    No modo manual:
        main_manual.py não usa esta função,
        porque quem define v e w és tu pelo teclado.

    Entrada:
        pose:
            pose atual do robô:
                [x, y, theta]

        target:
            waypoint alvo:
                [x_target, y_target]

        cfg:
            configuração do simulador.
            Usa:
                cfg.max_v
                cfg.max_w

    Saída:
        v:
            velocidade linear comandada [m/s]

        w:
            velocidade angular comandada [rad/s]

    Ideia:
        - calcular direção do alvo;
        - calcular erro angular;
        - andar mais depressa se o alvo estiver longe;
        - rodar para apontar para o alvo;
        - limitar velocidades máximas.
"""
def waypoint_controller(pose: np.ndarray, target: np.ndarray, cfg: SimConfig) -> Tuple[float, float]:
    # pose atual
    x, y, theta = pose

    # Calcula diferença em x/y entre o alvo e o robô.
    dx = target[0] - x
    dy = target[1] - y

    # Calcula distância euclidiana até ao alvo.
    distance = math.hypot(dx, dy)

    # Calcula a orientação global desejada para apontar para o alvo.
    desired_heading = math.atan2(dy, dx)

    # Calcula erro angular entre:
    #     direção desejada
    #     orientação atual do robô
    #
    # Depois normaliza para [-pi, pi].
    #
    # Exemplo:
    #     se desired_heading = 179°
    #     e theta = -179°
    #
    # A diferença bruta seria 358°,
    # mas a diferença real é -2°.
    #
    # normalize_angle corrige esse problema.
    heading_error = normalize_angle(desired_heading - theta)

     # Ganho proporcional da velocidade linear.
    #
    # Quanto maior k_v:
    #     mais agressivamente o robô avança para o alvo.
    #
    # v será proporcional à distância.
    k_v = 0.9

    # Ganho proporcional da velocidade angular.
    #
    # Quanto maior k_w:
    #     mais agressivamente o robô roda para apontar ao alvo.
    k_w = 2.3

    # Calcula velocidade linear proporcional à distância.
    #
    # k_v * distance:
    #     se está longe, pede velocidade maior;
    #     se está perto, pede velocidade menor.
    #
    # min(cfg.max_v, ...):
    #     limita a velocidade máxima.
    #
    # Exemplo:
    #     cfg.max_v = 0.45
    #
    # Mesmo que k_v * distance dê 3.0,
    # o robô só recebe 0.45 m/s.
    v = min(cfg.max_v, k_v * distance)

    # Reduz a velocidade linear se o robô não estiver apontado para o alvo.
    #
    # math.cos(heading_error):
    #
    #     se heading_error = 0:
    #         cos(0) = 1
    #         o robô anda normalmente.
    #
    #     se heading_error = 90°:
    #         cos(pi/2) = 0
    #         o robô não avança.
    #
    #     se heading_error = 180°:
    #         cos(pi) = -1
    #
    # max(0.0, ...) impede velocidade negativa.
    #
    # Resultado:
    #     se o robô está muito mal orientado,
    #     ele primeiro roda antes de avançar.
    v *= max(0.0, math.cos(heading_error))

    # Calcula velocidade angular proporcional ao erro angular.
    #
    # k_w * heading_error:
    #     se erro angular é positivo, roda num sentido;
    #     se erro angular é negativo, roda no outro.
    #
    # np.clip(valor, mínimo, máximo):
    #     limita o valor ao intervalo dado.
    #
    # Aqui limitamos:
    #     -cfg.max_w <= w <= cfg.max_w
    #
    # Isto evita velocidades angulares irreais.
    w = np.clip(k_w * heading_error, -cfg.max_w, cfg.max_w)
    return float(v), float(w)


"""
    Simula medições visuais tipo ArUco.

    Esta função representa, no micro-simulador, aquilo que no robô real virá
    da câmara + detector de ArUcos.

    Entrada:
        true_pose:
            pose verdadeira do robô:
                [x, y, theta]

            Usa-se a pose verdadeira porque estamos a simular o sensor.
            No mundo real, o sensor mede a realidade, não a estimativa do EKF.

        landmarks:
            landmarks verdadeiros do mundo:
                [id, x, y]

        cfg:
            configuração do simulador.

            Usa:
                cfg.sensor.max_range
                cfg.sensor.fov
                cfg.sensor.detection_probability
                cfg.sensor.outlier_probability
                cfg.noise.sim_sigma_range
                cfg.noise.sim_sigma_bearing

        rng:
            gerador aleatório usado para criar ruído.

    Saída:
        readings:
            lista de medições.

            Cada medição é um dicionário:
                {
                    "id": tag_id,
                    "range": distância_medida,
                    "bearing": ângulo_medido
                }

    Estas medições são passadas depois para:
        ekf.update(readings)

    O EKF usa:
        id:
            para saber qual landmark foi observado.

        range:
            distância relativa ao landmark.

        bearing:
            ângulo relativo ao landmark.
    """
def generate_visual_measurements(
    true_pose: np.ndarray,
    landmarks: np.ndarray,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> List[dict]:
  
    # pose
    x, y, theta = true_pose

    # Cria lista vazia onde serão guardadas as medições.
    #
    # No início, a câmara ainda não viu nada.
    readings: List[dict] = []


    # Percorre todos os landmarks verdadeiros do mundo.
    #
    # Extrai [id, x, y] de cada landmark lm     
    for lm in landmarks:
        tag_id = int(lm[0])
        lx, ly = float(lm[1]), float(lm[2])
        
        # Calcula vetor do robô até ao landmark em coordenadas globais.
        dx = lx - x
        dy = ly - y
        # Calcula distância verdadeira ao landmark, true_range = sqrt(dx^2 + dy^2)
        # Esta é a distância que a câmara mediria se não houvesse ruído.
        true_range = math.hypot(dx, dy)

        # Calcula bearing verdadeiro.
        #
        # math.atan2(dy, dx):
        #     ângulo global da linha robô → landmark.
        #
        # Subtraímos theta:
        #     para passar de ângulo global para ângulo relativo ao robô.
        #
        # Exemplo:
        #     se o robô está virado para o landmark,
        #     true_bearing ≈ 0.
        #
        # Se o landmark está à esquerda do robô:
        #     true_bearing > 0.
        #
        # Se está à direita:
        #     true_bearing < 0.
        true_bearing = normalize_angle(math.atan2(dy, dx) - theta)

        # Verifica se o landmark está visível para a câmara.
        #
        # Há duas condições:
        #
        # 1. distância dentro do alcance:
        #        true_range <= cfg.sensor.max_range
        #
        # 2. ângulo dentro do campo de visão:
        #        abs(true_bearing) <= cfg.sensor.fov / 2
        #
        # Se fov = 100 graus:
        #     a câmara vê de -50 graus a +50 graus.
        visible = true_range <= cfg.sensor.max_range and abs(true_bearing) <= cfg.sensor.fov / 2.0
        if not visible:
            continue

        # Mesmo que esteja visível, a deteção pode falhar.
        #
        # rng.random() devolve um número aleatório entre 0 e 1.
        #
        # cfg.sensor.detection_probability é a probabilidade de detetar.
        #
        # Exemplo:
        #     detection_probability = 0.90
        #
        # Se rng.random() der 0.95:
        #     0.95 > 0.90
        #     falhou a deteção.
        #
        # Isto simula:
        #     motion blur;
        #     ArUco parcialmente visível;
        #     iluminação má;
        #     detector a falhar.    
        if rng.random() > cfg.sensor.detection_probability:
            continue
        
        # Ruído aumenta com a distância.
        range_sigma = (cfg.noise.sim_sigma_range + 0.015 * true_range)

        # Ruído angular aumenta quando a tag está mais perto das bordas do FOV.
        bearing_difficulty = abs(true_bearing) / (cfg.sensor.fov / 2.0)
        bearing_sigma = (cfg.noise.sim_sigma_bearing * (1.0 + 2.0 * bearing_difficulty))

        measured_range = (true_range + cfg.noise.sim_range_bias + rng.normal(0.0, range_sigma))

        measured_bearing = (true_bearing + cfg.noise.sim_bearing_bias + rng.normal(0.0, bearing_sigma))
                        #
                        # cfg.sensor.outlier_probability define a probabilidade disso acontecer.
        if rng.random() < cfg.sensor.outlier_probability:
                            measured_range += rng.normal(0.7, 0.25)
                            measured_bearing += rng.normal(math.radians(20.0), math.radians(8.0))

        # Adiciona a medição final à lista de leituras.
        #
        # Cada leitura é um dicionário Python com:
        #
        #     "id":
        #         ID do landmark observado.
        #
        #     "range":
        #         distância medida.
        #
        #     "bearing":
        #         ângulo relativo medido.
        #
        # max(0.01, measured_range):
        #     impede distâncias negativas ou zero.
        #
        # Como o ruído é aleatório, teoricamente poderia dar uma distância
        # negativa se o landmark estivesse muito perto.
        #
        # Uma distância negativa não tem sentido físico,
        # então forçamos mínimo 0.01 m.
        #
        # normalize_angle(...):
        #     garante que o bearing fica em [-pi, pi].
        readings.append({
            "id": tag_id,
            "range": max(0.01, float(measured_range)),
            "bearing": normalize_angle(float(measured_bearing)),
        })

    return readings
