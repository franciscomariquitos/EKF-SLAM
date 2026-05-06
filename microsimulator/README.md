# EKF-SLAM Micro-simulator

Micro-simulador Python para validar EKF-SLAM antes de Gazebo/ROS 2/rosbags reais.

## Estrutura

```text
main.py          # ponto de entrada
config.py        # parâmetros do simulador, ruído e sensor
ekf_slam.py      # EKF-SLAM puro, independente de ROS
world.py         # waypoints, landmarks, controlador e sensor visual
simulation.py    # loop de simulação e Monte Carlo
evaluation.py    # métricas: RMSE, loop closure, landmark error
plotting.py      # gráficos
io_utils.py      # CSVs e impressão de métricas
utils.py         # funções matemáticas auxiliares
```

## Como correr

Dentro desta pasta:

```bash
python3 main.py
```

Sem abrir janelas, apenas guardar PNGs/CSVs:

```bash
python3 main.py --no-show --runs 50
```

Resultados em:

```text
microsim_results/
  single_run_map.png
  single_run_errors.png
  single_run_history.csv
  monte_carlo_metrics.csv
```

## Lógica

O simulador gera:

1. trajetória verdadeira do robô;
2. odometria com ruído e drift;
3. medições visuais tipo ArUco: `[range, bearing, id]`;
4. EKF-SLAM com prediction por incrementos de odometria e correction por landmarks.

O EKF está separado de ROS. Mais tarde, o mesmo `EKFSLAM` deve ser chamado por um `slam_node.py` ou por um `bag_runner.py`.

## Modo manual com teclado

Também existe um modo em que a trajetória não é definida por waypoints. O utilizador conduz o robô simulado com o teclado, de forma parecida ao `teleop_keyboard` do TurtleBot3.

```bash
python3 main_manual.py
```

Para limitar a duração e não abrir janelas no fim:

```bash
python3 main_manual.py --duration 60 --no-show
```

Controlos:

```text
W / seta cima      frente
S / seta baixo     trás
A / seta esquerda  rodar à esquerda
D / seta direita   rodar à direita
Z                  parar rotação
X ou espaço        parar tudo
+ / -              aumentar/diminuir velocidade linear
] / [              aumentar/diminuir velocidade angular
Q                  terminar e guardar resultados
```

Este modo continua a usar o mesmo EKF-SLAM. A diferença é apenas a origem dos comandos de movimento: em vez de um controlador automático por waypoints, os comandos vêm do teclado.
