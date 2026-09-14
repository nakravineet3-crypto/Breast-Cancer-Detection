"""
Loads the trained SpikingCNN (week3_model.pt) and runs one real test image
through it, recording every neuron's spike at every time step in the final
(output) layer and the layer just before it (fc1/lif4), then plots a raster
chart: x-axis = time step, y-axis = neuron index, a dot = that neuron fired.
"""
import sys
import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from spiking_model import SpikingCNN, NUM_STEPS
from breakhis_dataset import BreakHisDataset

PROJECT_ROOT = "."
IMG_SIZE = 64

model = SpikingCNN()
model.load_state_dict(torch.load("data/splits/week3_model.pt", map_location="cpu"))
model.eval()

transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
test_ds = BreakHisDataset("data/splits/split.csv", "test", transform=transform, project_root=PROJECT_ROOT)

# grab one real malignant and one real benign test image
mal_idx = test_ds.df[test_ds.df.label == "malignant"].index[0]
ben_idx = test_ds.df[test_ds.df.label == "benign"].index[0]

def run_and_record(img_tensor):
    x = model.tsgf(img_tensor.unsqueeze(0))
    mem1, mem2, mem3, mem4, memo = (
        model.lif1.init_leaky(), model.lif2.init_leaky(), model.lif3.init_leaky(),
        model.lif4.init_leaky(), model.lif_out.init_leaky(),
    )
    fc1_spikes, out_spikes = [], []
    with torch.no_grad():
        for t in range(NUM_STEPS):
            cur1 = model.pool(model.conv1(x)); spk1, mem1 = model.lif1(cur1, mem1)
            cur2 = model.pool(model.conv2(spk1)); spk2, mem2 = model.lif2(cur2, mem2)
            cur3 = model.pool(model.conv3(spk2)); spk3, mem3 = model.lif3(cur3, mem3)
            flat = model.gap(spk3).flatten(1)
            cur4 = model.fc1(flat); spk4, mem4 = model.lif4(cur4, mem4)
            curo = model.fc2(spk4); spko, memo = model.lif_out(curo, memo)
            fc1_spikes.append(spk4[0].numpy())
            out_spikes.append(spko[0].numpy())
    return fc1_spikes, out_spikes


fig, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

for col, (idx, title) in enumerate([(ben_idx, "Benign test image"), (mal_idx, "Malignant test image")]):
    img, label = test_ds[idx]
    fc1_spikes, out_spikes = run_and_record(img)

    ax_top = axes[0, col]
    for neuron in range(32):
        times = [t for t in range(NUM_STEPS) if fc1_spikes[t][neuron] > 0]
        ax_top.scatter(times, [neuron] * len(times), s=10, color="tab:blue")
    ax_top.set_title(f"{title} — fc1 layer (32 neurons)")
    ax_top.set_xlabel("time step")
    ax_top.set_ylabel("neuron index")
    ax_top.set_xlim(-0.5, NUM_STEPS - 0.5)

    ax_bot = axes[1, col]
    labels_out = ["benign\nneuron", "malignant\nneuron"]
    for neuron in range(2):
        times = [t for t in range(NUM_STEPS) if out_spikes[t][neuron] > 0]
        ax_bot.scatter(times, [neuron] * len(times), s=30, color="tab:red")
    ax_bot.set_yticks([0, 1]); ax_bot.set_yticklabels(labels_out)
    ax_bot.set_xlabel("time step")
    ax_bot.set_title(f"Output layer — true label: {['benign','malignant'][label]}")
    ax_bot.set_xlim(-0.5, NUM_STEPS - 0.5)

plt.tight_layout()
plt.savefig("data/splits/spike_raster.png", dpi=130)
print("Saved data/splits/spike_raster.png")
