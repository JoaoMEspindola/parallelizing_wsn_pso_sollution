import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ===============================================================
# 1) LER CSVs E TRATAR LINHA "END"
# ===============================================================

def load_timeline(path, label):
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        # Localiza linha END
        end_row = df[df["time_ms"] == "END"]
        if len(end_row) == 0:
            print(f"[WARN] {path} não tem linha END. Será ignorado.")
            return None

        total_time = float(end_row["fitness"].iloc[0])

        df = df[df["time_ms"] != "END"].copy()

        df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
        df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")
        df = df.dropna()

        # >>> CONVERTE TODA A ESCALA PARA SEGUNDOS <<<
        df["time_ms"] = df["time_ms"] / 1000.0
        total_time = total_time / 1000.0

        df["label"] = label
        df["total_time"] = total_time


        return df

    except Exception as e:
        print(f"[ERRO] {path}: {e}")
        return None


# ===============================================================
# 2) EXPANDIR DEGRAU ATÉ O FIM DA EXECUÇÃO REAL
# ===============================================================

def expand_to_end(df):
    """ Expande até o próprio total_time (fim real da execução). """
    times = df["time_ms"].values
    fits = df["fitness"].values
    t_end = df["total_time"].iloc[0]

    new_t = []
    new_f = []

    for i in range(len(times)):
        t0 = times[i]
        f0 = fits[i]

        new_t.append(t0)
        new_f.append(f0)

        # até a próxima mudança ou até t_end
        t1 = times[i+1] if i+1 < len(times) else t_end

        new_t.append(t1)
        new_f.append(f0)

    return pd.DataFrame({
        "time_ms": new_t,
        "fitness": new_f,
        "label": df["label"].iloc[0],
        "total_time": t_end
    })


# ===============================================================
# 3) EXTENDER ALÉM DO FIM REAL (SERRILHADO E OPACO)
# ===============================================================

def extend_dashed(df, global_end):
    """ Cria uma linha serrilhada mantendo o último fitness. """

    end_real = df["total_time"].iloc[0]
    final_fit = df["fitness"].iloc[-1]

    if end_real >= global_end:
        return None  # não precisa extender

    return pd.DataFrame({
        "time_ms": [end_real, global_end],
        "fitness": [final_fit, final_fit],
        "label": df["label"].iloc[0],
        "style": "dashed"
    })


# ===============================================================
# 4) PLOTAR
# ===============================================================

def plot_gbest_timelines(cpu_path, gpu_path, gpu_block_path):
    df_cpu = load_timeline(cpu_path, "CPU")
    df_gpu = load_timeline(gpu_path, "GPU Thread")
    df_block = load_timeline(gpu_block_path, "GPU Block")

    dfs = [d for d in [df_cpu, df_gpu, df_block] if d is not None]

    if not dfs:
        print("[ERRO] Nenhum arquivo válido.")
        return

    GLOBAL_END = max(df["total_time"].iloc[0] for df in dfs)

    plt.figure(figsize=(13, 7))

    colors = {
        "CPU": "tab:blue",
        "GPU Thread": "tab:green",
        "GPU Block": "tab:red"
    }

    # ======= PLOT PRINCIPAL =======
    for df in dfs:
        label = df["label"].iloc[0]

        # (1) Expandir linha real
        step_df = expand_to_end(df)

        plt.plot(
            step_df["time_ms"], step_df["fitness"],
            label=label,
            linewidth=2,
            color=colors[label]
        )

        # (2) Extender serrilhado
        dashed_df = extend_dashed(step_df, GLOBAL_END)
        if dashed_df is not None:
            plt.plot(
                dashed_df["time_ms"], dashed_df["fitness"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.4,
                color=colors[label]
            )

        # (3) Marca o ponto final
        final_t = step_df["total_time"].iloc[0]
        final_f = step_df["fitness"].iloc[-1]

        plt.scatter([final_t], [final_f], color=colors[label], s=40)
        plt.text(
        GLOBAL_END + GLOBAL_END*0.01,
        final_f,
        f"{label}: {final_f:.2f}  ({final_t:.2f}s)",
        color=colors[label],
        fontsize=10,
        va="center"
    )


    plt.title("Evolução do Melhor Fitness (GBest) ao longo do tempo (Rede 3)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Fitness (lifetime)")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, GLOBAL_END * 1.15)
    plt.legend()
    plt.tight_layout()

    plt.savefig("gbest_timeline_net3.png", dpi=300)
    plt.show()


def plot_gpu_vs_gpu_target(gpu_path, gpu_block_path, target_fitness):
    """
    Plota apenas GPU Thread e GPU Block com a linha horizontal marcando o fitness alvo.
    Mantém todo o estilo visual do gráfico principal.
    """

    df_gpu = load_timeline(gpu_path, "GPU Thread")
    df_block = load_timeline(gpu_block_path, "GPU Block")

    dfs = [d for d in [df_gpu, df_block] if d is not None]

    if not dfs:
        print("[ERRO] Nenhum arquivo válido das GPUs.")
        return

    GLOBAL_END = max(df["total_time"].iloc[0] for df in dfs)

    plt.figure(figsize=(13, 7))

    colors = {
        "GPU Thread": "tab:green",
        "GPU Block": "tab:red"
    }

    # =============== PLOT PRINCIPAL ===============
    for df in dfs:
        label = df["label"].iloc[0]

        # (1) Expandir linha real até o fim real
        step_df = expand_to_end(df)

        # detectar atingimento do target
        hit = step_df[step_df["fitness"] >= target_fitness]
        if not hit.empty:
            t_hit = hit["time_ms"].iloc[0]
            legend_suffix = f"{t_hit:.2f}s"
        else:
            legend_suffix = "-"

        # ajusta o rótulo para a legenda
        legend_label = f"{label} – {legend_suffix} (Target {target_fitness})"

        plt.plot(
            step_df["time_ms"], step_df["fitness"],
            label=legend_label,
            linewidth=2,
            color=colors[label]
        )

        # (2) Extender serrilhado até o fim GLOBAL
        dashed_df = extend_dashed(step_df, GLOBAL_END)
        if dashed_df is not None:
            plt.plot(
                dashed_df["time_ms"], dashed_df["fitness"],
                linestyle="--",
                linewidth=1.5,
                alpha=0.4,
                color=colors[label]
            )

        # (3) Ponto final + texto igual ao estilo original
        final_t = step_df["total_time"].iloc[0]
        final_f = step_df["fitness"].iloc[-1]

        plt.scatter([final_t], [final_f], color=colors[label], s=40)        

    # ================================
    # LINHA HORIZONTAL DO FITNESS ALVO
    # ================================
    plt.axhline(target_fitness, linestyle="--", color="black", linewidth=1.5, alpha=0.7)

    # ================================
    # CONFIGURAÇÕES FINAIS
    # ================================
    plt.title("Comparação GPU Thread vs GPU Block com Target Fitness (Rede 3)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Fitness")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, GLOBAL_END * 1.15)
    plt.legend()
    plt.tight_layout()

    plt.savefig("gpu_vs_gpu_target_r3.png", dpi=300)
    plt.show()


# ===============================================================

if __name__ == "__main__":
    # plot_gbest_timelines(
    #     "defaultPlotting/bests/gbest_timeline_cpu_bestr3.csv",
    #     "defaultPlotting/bests/gbest_timeline_gpu_bestr3.csv",
    #     "defaultPlotting/bests/gbest_timeline_gpu_block_bestr3.csv"
    # )

    plot_gpu_vs_gpu_target(
        "targetPlotting/r3/gbest_timeline_gpu2.csv",
        "targetPlotting/r3/gbest_timeline_gpu_block2.csv",
        target_fitness=2000
    )
