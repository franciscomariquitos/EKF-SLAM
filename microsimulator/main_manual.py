"""
main_manual.py
--------------
Execução:
    python3 main_manual.py
    python3 main_manual.py --live-plot
    python3 main_manual.py --live-plot --no-show

Implementação:
    1. ler opções do terminal;
    2. criar a configuração;
    3. chamar a simulação manual;
    4. guardar resultados;
    5. fazer gráficos finais.

"""

# Esta linha permite usar "type hints" mais modernas.
# Type hints são anotações do tipo:
#     x: int
#     nome: str
#     função(...) -> None
#
# Não é obrigatório para o programa funcionar,
# mas deixa o código mais claro.
from __future__ import annotations

# argparse é uma biblioteca standard do Python.
# Serve para ler argumentos escritos no terminal.
#
# Exemplo:
#     python3 main_manual.py --duration 60 --live-plot
#
# Aqui, argparse permite ler:
#     --duration
#     --seed
#     --outdir
#     --no-show
#     --live-plot
#     --gate
import argparse

# Path é uma forma moderna de lidar com caminhos de ficheiros/pastas.
#
# Em vez de manipular strings como:
#     "pasta/ficheiro.csv"
#
# usamos:
#     Path("pasta") / "ficheiro.csv"
#
# É mais limpo e funciona melhor entre sistemas operativos.
from pathlib import Path

# Importa a classe SimConfig do ficheiro config.py.
#
# SimConfig guarda os parâmetros do simulador:
#     dt
#     velocidade máxima
#     ruído da odometria
#     ruído da câmara
#     field of view
#     Mahalanobis gate
#     etc.
from config import SimConfig

# Importa funções do ficheiro io_utils.py.
#
# print_metrics:
#     imprime as métricas no terminal.
#
# save_history_csv:
#     guarda o histórico da simulação num ficheiro CSV.
from io_utils import print_metrics, save_history_csv

# Importa a função principal do modo manual.
#
# Esta função está em manual_teleop.py.
#
# É ela que:
#     lê teclas;
#     move o robô;
#     gera odometria com ruído;
#     chama o EKF;
#     atualiza o live plot;
#     guarda histórico.
from manual_teleop import run_manual_teleop_simulation

# Importa funções para criar gráficos finais depois da simulação acabar.
#
# plot_map:
#     desenha ground truth, odometria, EKF e landmarks.
#
# plot_errors:
#     desenha o erro da odometria e do EKF ao longo do tempo.
from plotting import plot_errors, plot_map


"""
    Esta função cria o parser dos argumentos de terminal, definindo quais são as opções que se pode escrever na inicialização
    Ex:
        python3 main_manual.py --live-plot --duration 60

    Retorna:
        um objeto ArgumentParser configurado.
"""
def build_arg_parser() -> argparse.ArgumentParser:

     # Cria o parser.
    #
    # description é o texto que aparece qd se faz:
    #     python3 main_manual.py --help
    parser = argparse.ArgumentParser(description="Interactive EKF-SLAM micro-simulator with keyboard teleop.")
     # Argumento --seed
    #
    # A seed controla a aleatoriedade do simulador.
    #
    # Se usares sempre a mesma seed, o ruído gerado será igual.
    # Isso é muito útil para debug e para repetir experiências.
    #
    # Exemplo:
    #     python3 main_manual.py --seed 10
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    
    # Argumento --duration
    #
    # Define a duração máxima da simulação manual, em segundos.
    #
    # Exemplo:
    #     python3 main_manual.py --duration 60
    #
    # Se não carregares em Q antes, a simulação acaba aos 60 s. Default é 120s
    parser.add_argument("--duration", type=float, default=120.0, help="Maximum teleop duration [s].")

    # Argumento --outdir
    #
    # Define a pasta onde os resultados vão ser guardados.
    #
    # Exemplo:
    #     python3 main_manual.py --outdir resultados_teste_1
    parser.add_argument("--outdir", type=str, default="microsim_manual_results", help="Output folder.")

     # Argumento --no-show
    #
    # Isto é um booleano.
    # --no-show, fica True.
    # Se não, fica False.
    #
    # Serve para não abrir janelas Matplotlib no fim.
    # Mesmo assim, os PNGs são guardados.
    #
    # Exemplo:
    #     python3 main_manual.py --no-show
    parser.add_argument("--no-show", action="store_true", help="Only save plots; do not open matplotlib windows after finishing.")

    # Argumento --live-plot
    #
    # Abre uma janela live enquanto conduzes.
    #
    # Exemplo:
    #     python3 main_manual.py --live-plot
    #
    # Atenção:
    #     as teclas continuam a ser lidas pelo terminal;
    #     a janela Matplotlib é só visualização.
    parser.add_argument("--live-plot", action="store_true", help="Open a live matplotlib window showing the robot moving during teleop.")

     # Argumento --gate
    #
    # Define o limiar de Mahalanobis usado para rejeitar outliers.
    #
    # 9.21 é aproximadamente o valor chi-square para uma medição 2D
    # com 99% de confiança.
    #
    # Se o valor for menor:
    #     rejeita mais medições.
    #
    # Se o valor for maior:
    #     aceita mais medições.
    parser.add_argument("--gate", type=float, default=9.21, help="Mahalanobis gate for visual measurements.")
    return parser


def main() -> None:
    """
    Função principal do programa.

    Ordem lógica:
        1. ler argumentos do terminal;
        2. criar pasta de saída;
        3. criar configuração;
        4. correr simulação manual;
        5. imprimir métricas;
        6. guardar CSV;
        7. gerar gráficos finais.
    """
    args = build_arg_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = SimConfig(seed=args.seed, mahalanobis_gate=args.gate)

    result = run_manual_teleop_simulation(cfg, max_duration_s=args.duration, live_plot=args.live_plot)
    print_metrics("Manual teleop simulation metrics", result["metrics"])
    save_history_csv(result, outdir)
    plot_map(result, outdir, show=not args.no_show)
    plot_errors(result, outdir, show=not args.no_show)

    print(f"\nSaved results to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
