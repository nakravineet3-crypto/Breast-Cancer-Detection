"""
Week 4 — full MASNN: TSGF -> spiking conv layers -> multi-dimensional attention
(CBAM: channel + spatial) -> spiking recurrent layer -> FC spiking layer -> output,
matching the order in the paper's Figure 2. Also adds temporal attention (a
learnable weight per time step) when combining the 25 time steps into a final
decision, instead of the Week 3 model's plain uniform sum.
"""
import torch
import torch.nn as nn
import snntorch as snn

from tsgf import TrainableSelfGuidedFilter
from attention import CBAM

NUM_STEPS = 25
THRESHOLD = 0.3  # same empirically-tuned value from Week 3


class MASNN(nn.Module):
    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.tsgf = TrainableSelfGuidedFilter(radius=4)

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, threshold=THRESHOLD)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=THRESHOLD)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Multi-dimensional attention block: channel + spatial (CBAM), applied
        # once, on the deepest/most semantically-rich conv layer's output —
        # matches the single "Multi-Dimensional Attention Block" placement in
        # the paper's Fig. 2, positioned after the spiking conv layers.
        self.cbam = CBAM(channels=64)
        self.lif3 = snn.Leaky(beta=beta, threshold=THRESHOLD)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32)

        # Spiking recurrent layer: a LIF neuron whose own previous output
        # spike feeds back in as extra input each step (learn_recurrent=True
        # makes that feedback connection itself trainable, not fixed).
        self.rlif = snn.RLeaky(beta=beta, threshold=THRESHOLD, linear_features=32, learn_recurrent=True)

        self.fc2 = nn.Linear(32, 2)
        self.lif_out = snn.Leaky(beta=beta, threshold=THRESHOLD)

        # Temporal attention: one learnable weight per time step, softmax-
        # normalized so they sum to 1 — replaces Week 3's uniform spike sum
        # with a learned weighting of which time steps to trust more.
        self.time_weight = nn.Parameter(torch.ones(NUM_STEPS))

    def forward(self, x):
        x = self.tsgf(x)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_r, mem_r = self.rlif.init_rleaky()
        mem_out = self.lif_out.init_leaky()

        outputs_per_step = []

        for t in range(NUM_STEPS):
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.conv3(spk2)
            cur3 = self.cbam(cur3)          # attention reweights channels + spatial locations here
            cur3 = self.pool(cur3)
            spk3, mem3 = self.lif3(cur3, mem3)

            flat = self.gap(spk3).flatten(1)
            cur4 = self.fc1(flat)
            spk_r, mem_r = self.rlif(cur4, spk_r, mem_r)

            cur_out = self.fc2(spk_r)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            outputs_per_step.append(spk_out)

        stacked = torch.stack(outputs_per_step, dim=0)             # (NUM_STEPS, batch, 2)
        weights = torch.softmax(self.time_weight, dim=0).view(-1, 1, 1)  # (NUM_STEPS, 1, 1), sums to 1
        # scaled by NUM_STEPS so output magnitude is comparable to Week 3's plain spike count
        weighted_sum = (stacked * weights).sum(dim=0) * NUM_STEPS
        return weighted_sum


def create_healthy_masnn(probe_batch: torch.Tensor, max_attempts: int = 10) -> "MASNN":
    """Same 'born-dead network' safeguard from Week 3 — now for the full MASNN."""
    for attempt in range(1, max_attempts + 1):
        model = MASNN()
        with torch.no_grad():
            out = model(probe_batch)
        total = out.sum().item()
        if total > 0:
            print(f"[create_healthy_masnn] attempt {attempt}: alive (sum={total:.2f}) — using this init")
            return model
        print(f"[create_healthy_masnn] attempt {attempt}: dead — retrying")
    raise RuntimeError(f"Could not find a non-dead initialization in {max_attempts} attempts")
