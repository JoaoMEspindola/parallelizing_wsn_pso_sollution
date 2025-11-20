# visualize_network_final.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.patches import Circle

# CONFIG ------------------------------------------------
CSV_PATH = "gpu_block_network.csv"   # troque conforme necessário
SHOW_RADII = True
RADII_ALPHA = 0.18        # transparência dos círculos dos gateways
RADII_EDGESTYLE = (0, (3,3))  # dash style (dash, gap)
SAVE_FIG = False
OUT_IMAGE = "network_map.png"
# -------------------------------------------------------

CLUSTER_COLORS = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
]

def load_and_prepare(path):
    df = pd.read_csv(path)

    # normalize column names in case of spaces etc.
    df.columns = [c.strip() for c in df.columns]

    # convert types safely
    for col in ["id","from","to","nextHop","cluster"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # clusterRadius numeric
    if "clusterRadius" in df.columns:
        df["clusterRadius"] = pd.to_numeric(df["clusterRadius"], errors="coerce")
    else:
        df["clusterRadius"] = np.nan

    return df

def get_color_for_cluster(cluster):
    if pd.isna(cluster) or int(cluster) < 0:
        return "#bdbdbd"   # sensores sem cluster = cinza claro
    return CLUSTER_COLORS[int(cluster) % len(CLUSTER_COLORS)]

def plot_network(df, title="Network"):

    # -------------------------------
    # Filtrar nós e arestas corretamente
    # -------------------------------
    nodes = df[df["recordType"] == "node"].copy()
    edges = df[df["recordType"] == "edge"].copy()

    # converter ids para numeric
    nodes["id"] = pd.to_numeric(nodes["id"], errors="coerce")
    edges["from"] = pd.to_numeric(edges["from"], errors="coerce")
    edges["to"] = pd.to_numeric(edges["to"], errors="coerce")

    # Remover arestas com endpoints inválidos
    valid_ids = set(nodes["id"].dropna().astype(int))
    edges = edges[
        edges["from"].isin(valid_ids) &
        edges["to"].isin(valid_ids)
    ].copy()

    # -------------------------------
    # Plot
    # -------------------------------
    fig, ax = plt.subplots(figsize=(10,10))
    
    # Draw Nodes
    gw = nodes[nodes["isGateway"] == 1]
    sn = nodes[nodes["isGateway"] == 0]

    ax.scatter(sn["x"], sn["y"], c="blue", s=9, alpha=0.30, label="Sensors")
    ax.scatter(gw["x"], gw["y"], c="red", s=18, alpha=0.75, label="Gateways")

    # Draw Base Station (id max)
    bs = nodes.iloc[nodes["id"].idxmax()]
    ax.scatter(bs["x"], bs["y"], c="black", marker="X", s=70, label="Base Station", zorder=5)

    # -------------------------------
    # Draw Edges (safe version)
    # -------------------------------
    # -------------------------------
# Draw Edges (by type)
# -------------------------------
    for _, e in edges.iterrows():
        a, b = int(e["from"]), int(e["to"])
        et = str(e["edgeType"]) if not pd.isna(e["edgeType"]) else ""

        na = nodes[nodes["id"] == a]
        nb = nodes[nodes["id"] == b]

        if na.empty or nb.empty:
            continue

        x1 = float(na.iloc[0]["x"])
        y1 = float(na.iloc[0]["y"])
        x2 = float(nb.iloc[0]["x"])
        y2 = float(nb.iloc[0]["y"])

        # Sensor → Gateway edges
        if et == "Sensor-Gateway":
            ax.plot([x1, x2], [y1, y2],
                    color="black", linewidth=0.35, alpha=0.06, zorder=0)

        # Gateway → Gateway edges
        elif et == "Gateway-Gateway":
            ax.plot([x1, x2], [y1, y2],
                    color="#FF8800", linewidth=0.8, alpha=0.25, zorder=1)  # laranja mais visível

        # Gateway → BS edges
        elif et == "Gateway-BS":
            ax.plot([x1, x2], [y1, y2],
                    color="#00AAFF", linewidth=1.2, alpha=0.45, zorder=2)  # azul claro

        # fallback (deixar igual sensor-gateway)
        else:
            ax.plot([x1, x2], [y1, y2],
                    color="black", linewidth=0.4, alpha=0.08, zorder=0)


    # -------------------------------
    # Draw cluster radii
    # -------------------------------
    if "clusterRadius" in nodes.columns:
        for _, row in gw.iterrows():
            if not pd.isna(row["clusterRadius"]):
                circ = plt.Circle(
                    (row["x"], row["y"]),
                    row["clusterRadius"],
                    color="red",
                    alpha=0.06,
                    fill=True
                )
                ax.add_patch(circ)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    df = load_and_prepare(CSV_PATH)
    plot_network(df, title="Network (with cluster radii)")
