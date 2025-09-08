import os
import sys
sys.path.insert(0, os.getcwd())
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datasets.PoseIndustrial6D.dataloader_20m import PoseDataset2
from tqdm import tqdm

distance_by_class = defaultdict(list)

dataset = PoseDataset2(mode="test")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for data in tqdm(loader):
    # unpack data
    _, _, _, _, _, _, _, _, _, _, rt, idx = data

    # extrair translação da matriz 4x4 (última coluna)
    t = rt[0, :3, 3].numpy()
    distance = np.linalg.norm(t)

    # distâncias acima de 20m já foram excluídas no dataset
    class_id = str(idx.item())
    distance_by_class[class_id].append(distance)

# Mesmos bins e nomes de classe
bins = [0, 5, 10, 15, 20]
bin_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]
class_names = ["Industrial Drums", "Boxes", "Forklift Slots", "Extinguishers", "Forklifts", "People", "Toolboxes"]

num_classes = len(class_names)
cols = 4
rows = 2

fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
axes = axes.flatten()

for i, name in enumerate(class_names):
    cls_id = str(i)
    distances = distance_by_class.get(cls_id, [])
    ax = axes[i]

    if distances:
        counts, _ = np.histogram(distances, bins=bins)
        bars = ax.hist(distances, bins=bins, color='skyblue', edgecolor='black', rwidth=0.9)

        for center, count in zip(bin_centers, counts):
            if count > 0:
                ax.text(center, count + 0.5, str(int(count)), ha='center', va='bottom', fontsize=8)

        ax.set_title(f'{name}')

    ax.set_xticks(bin_centers)
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Distance to origin(m)")
    ax.set_ylabel("Number of detections")
    ax.grid(True, linestyle='--', alpha=0.5)

"""# Esconde subplots extras se houver
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])"""

summary_ax = axes[len(class_names)]
summary_ax.axis('off')  # Oculta o eixo

summary_lines = [r"$\bf{Total\ Annotations\ per\ Class}$"]

# Adiciona as linhas com contagens
total_annotations = 0
for i, name in enumerate(class_names):
    cls_id = str(i)
    count = len(distance_by_class.get(cls_id, []))
    total_annotations += count
    summary_lines.append(f"{name}: {count}")

# Adiciona linha com o total geral
summary_lines.append("")
summary_lines.append(rf"$\bf{{Total\ Annotations:}}$ {total_annotations}")

# Junta as linhas com espaçamento normal
summary_text = "\n".join(summary_lines)

# Exibe o texto no eixo
summary_ax.text(
    0.01, 0.98, summary_text,
    va='top', ha='left',
    fontsize=10,
    linespacing=2
)

fig.tight_layout()
plt.show()

