"""
utils.py
--------

Este ficheiro contém funções matemáticas pequenas usadas em vários pontos
do micro-simulador.

Estas funções aparecem em:
    - simulation.py
    - manual_teleop.py
    - world.py
    - ekf_slam.py
    - evaluation.py

No contexto do projeto:
    - pose_step(...) simula o movimento do TurtleBot;
    - odometry_increment(...) imita o fluxo real do tópico /odom;
    - normalize_angle(...) evita problemas com ângulos;
    - rmse(...) calcula métricas para comparar odometria vs EKF.
"""

# Permite type hints modernas.
#
# Isto não muda a execução matemática do código.
# Serve apenas para melhorar compatibilidade e legibilidade.
from __future__ import annotations

# Importa a biblioteca math.
#
# math contém funções matemáticas básicas:
#     math.pi
#     math.cos(...)
#     math.sin(...)
#     math.atan2(...)
#     math.hypot(...)
#
# Usamos isto para geometria 2D do robô.
import math

# Importa Tuple para type hints.
#
# Tuple[float, float, float]
# significa:
#     a função devolve exatamente três valores float.
#
# No caso deste ficheiro, isso é usado em odometry_increment(...),
# que devolve:
#     delta_rot1
#     delta_trans
#     delta_rot2
from typing import Tuple

# Importa NumPy.
#
# NumPy é usado para:
#     representar poses como arrays;
#     criar arrays novos;
#     calcular médias;
#     calcular raízes quadradas;
#     converter listas para arrays.
import numpy as np



"""
    Normaliza um ângulo para o intervalo [-pi, pi].

    Isto significa que qualquer ângulo fica entre:
        -pi  e  pi

    Em graus:
        -180º e 180º

    Porquê isto é necessário?

    Porque em robótica móvel, ângulos dão a volta.

    Exemplo:
        190º é equivalente a -170º.

    Outro exemplo importante:
        +179º e -179º parecem muito diferentes numericamente,
        mas fisicamente estão separados só por 2º.

    Sem esta função, o EKF poderia pensar que o erro angular é enorme
    quando na realidade é pequeno.

    Entrada:
        angle:
            ângulo em radianos.

    Saída:
        ângulo equivalente em radianos dentro de [-pi, pi].
    """
def normalize_angle(angle: float) -> float:
    # math.pi é o valor de pi.
    #
    # A expressão:
    #
    #     (angle + pi) % (2*pi) - pi
    #
    # faz o "wrap" do ângulo.
    #
    # Explicação:
    #
    # 1. angle + pi
    #       desloca o intervalo desejado.
    #
    # 2. % (2*pi)
    #       aplica módulo por uma volta completa.
    #       Uma volta completa tem 2*pi radianos.
    #
    # 3. - pi
    #       volta a deslocar para o intervalo [-pi, pi].
    #
    # Exemplo aproximado:
    #
    #     angle = 3*pi
    #
    #     3*pi é equivalente a pi.
    #
    # A função devolve pi ou -pi dependendo da convenção numérica.
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


"""
    Integra o movimento de um robô tipo uniciclo/diferencial.

    Esta função simula como a pose do robô muda depois de um pequeno intervalo
    de tempo dt.

    É usada para:
        - atualizar a pose verdadeira no simulador;
        - atualizar a pose de odometria com ruído.

    Modelo usado:
        pose = [x, y, theta]

        v:
            velocidade linear [m/s]

        w:
            velocidade angular [rad/s]

        dt:
            intervalo de tempo [s]

    O TurtleBot3 é aproximadamente um robô diferencial.
    Um robô diferencial pode ser representado no plano como modelo uniciclo:

        x_dot     = v cos(theta)
        y_dot     = v sin(theta)
        theta_dot = w

    A versão discreta simples é:
        theta_new = theta + w dt
        x_new     = x + v dt cos(theta_new)
        y_new     = y + v dt sin(theta_new)

"""
def pose_step(pose: np.ndarray, v: float, w: float, dt: float) -> np.ndarray:
    # Desempacota a pose atual.
    #
    # pose é um array com três elementos:
    #     pose[0] = x
    #     pose[1] = y
    #     pose[2] = theta
    #
    # Depois desta linha:
    #     x recebe pose[0]
    #     y recebe pose[1]
    #     theta recebe pose[2]
    x, y, theta = pose

    # Atualiza a orientação.
    #
    # w * dt é a rotação acumulada durante o intervalo dt.
    #
    # Exemplo:
    #     w = 1 rad/s
    #     dt = 0.05 s
    #
    #     rotação = 1 * 0.05 = 0.05 rad
    #
    # normalize_angle garante que theta_new fica entre -pi e pi.
    theta_new = normalize_angle(theta + w * dt)

    # Atualiza x.
    #
    # v * dt é a distância percorrida durante este passo.
    #
    # math.cos(theta_new) projeta essa distância no eixo x global.
    #
    # Exemplo:
    #     se theta_new = 0:
    #         cos(0) = 1
    #         todo o movimento vai para x.
    #
    #     se theta_new = pi/2:
    #         cos(pi/2) = 0
    x_new = x + v * dt * math.cos(theta_new)


    # Atualiza y.
    #
    # math.sin(theta_new) projeta a distância no eixo y global.
    #
    # Exemplo:
    #     se theta_new = 0:
    #         sin(0) = 0
    #         não há movimento em y.
    #
    #     se theta_new = pi/2:
    #         sin(pi/2) = 1
    #         todo o movimento vai para y.
    y_new = y + v * dt * math.sin(theta_new)

    return np.array([x_new, y_new, theta_new], dtype=float)


"""
    Converte duas poses consecutivas de odometria no modelo clássico:

        u = (delta_rot1, delta_trans, delta_rot2)

    Isto é importante porque, no robô real, o EKF vai receber poses do tópico:

        /odom

    O tópico /odom não te dá diretamente:
        delta_rot1
        delta_trans
        delta_rot2

    Ele dá uma pose ao longo do tempo.

    Portanto, para usar o modelo probabilístico de odometria no EKF,
    calculamos o incremento entre duas poses consecutivas:

        prev_odom → curr_odom

    A decomposição é:

        1. delta_rot1:
            rotação inicial para apontar na direção do deslocamento;

        2. delta_trans:
            distância percorrida;

        3. delta_rot2:
            rotação final para chegar à orientação atual.

    Entrada:
        prev_odom:
            pose de odometria anterior:
                [x1, y1, theta1]

        curr_odom:
            pose de odometria atual:
                [x2, y2, theta2]

    Saída:
        delta_rot1:
            primeira rotação [rad]

        delta_trans:
            translação [m]

        delta_rot2:
            segunda rotação [rad]
    """
def odometry_increment(prev_odom: np.ndarray, curr_odom: np.ndarray) -> Tuple[float, float, float]:

    # Desempacota a pose anterior da odometria.
    #
    # prev_odom tem:
    #     x1  = posição x anterior
    #     y1  = posição y anterior
    #     th1 = orientação anterior
    x1, y1, th1 = prev_odom


    # Desempacota a pose atual da odometria.
    #
    # curr_odom tem:
    #     x2  = posição x atual
    #     y2  = posição y atual
    #     th2 = orientação atual
    x2, y2, th2 = curr_odom

    # Calcula deslocamento em x e y entre as duas poses.
    dx = x2 - x1
    dy = y2 - y1

    # Calcula distância percorrida no plano.
    #
    # math.hypot(dx, dy) faz:
    #     sqrt(dx^2 + dy^2)
    #
    # Isto é delta_trans.
    delta_trans = math.hypot(dx, dy)

    # Se a translação for praticamente zero,
    # não conseguimos definir uma direção de movimento fiável.
    #
    # Exemplo:
    #     dx = 0
    #     dy = 0
    #
    # atan2(0, 0) é matematicamente indefinido.
    #
    # Então, neste caso, definimos delta_rot1 = 0.
    #
    # 1e-12 é um valor muito pequeno usado como tolerância numérica.
    if delta_trans < 1e-12:
        delta_rot1 = 0.0
    else:
        # math.atan2(dy, dx) dá a direção global do deslocamento.
        #
        # Subtraímos th1 porque queremos saber quanto o robô teve de rodar
        # relativamente à orientação anterior.
        #
        # normalize_angle garante que o resultado fica em [-pi, pi].
        #
        # Exemplo:
        #     se o robô estava orientado para 0 rad,
        #     e se deslocou para a direção pi/2,
        #
        #     delta_rot1 = pi/2 - 0 = pi/2
        delta_rot1 = normalize_angle(math.atan2(dy, dx) - th1)
    
    # Calcula a segunda rotação.
    #
    # A orientação final th2 deve ser:
    #
    #     th2 = th1 + delta_rot1 + delta_rot2
    #
    # Rearranjando:
    #
    #     delta_rot2 = th2 - th1 - delta_rot1
    #
    # Também normalizamos para evitar saltos de ângulo
    delta_rot2 = normalize_angle(th2 - th1 - delta_rot1)
    return delta_rot1, delta_trans, delta_rot2

"""
    Calcula o RMSE de um vetor de erros.

    Fórmula:

        RMSE = sqrt(mean(error^2))

    Usado para comparar:
        - erro da odometria;
        - erro do EKF-SLAM;
        - erro dos landmarks.
    """
def rmse(values: np.ndarray) -> float:
   
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values**2)))
