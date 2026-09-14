"""
First spiking classifier — Week 3. Mirrors BaselineCNN's structure (TSGF ->
3 conv blocks -> classifier head) but every conv block's activation is a real
LIF spiking neuron (snn.Leaky) instead of ReLU, and the whole thing runs for
multiple time steps per image instead of a single instantaneous pass.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

from tsgf import TrainableSelfGuidedFilter

NUM_STEPS = 25  # how many time steps each image is "shown" for


class SpikingCNN(nn.Module):
    def __init__(self, beta: float = 0.9):
        super().__init__()
        self.tsgf = TrainableSelfGuidedFilter(radius=4)

        # threshold=0.3 (not snnTorch's default of 1.0) — tuned empirically:
        # with the default threshold, current magnitude shrinks with depth
        # (each layer's input comes from an already-sparse, mostly-0/1 spike
        # map), so deeper layers effectively never fire at threshold=1.0 —
        # verified by measuring per-layer spike rates directly. threshold=0.3
        # keeps every layer meaningfully active (16-55% firing rate) without
        # saturating to "fires on literally every step" either.
        thresh = 0.3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.lif1 = snn.Leaky(beta=beta, threshold=thresh)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2 = snn.Leaky(beta=beta, threshold=thresh)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif3 = snn.Leaky(beta=beta, threshold=thresh)

        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(64, 32)
        self.lif4 = snn.Leaky(beta=beta, threshold=thresh)
        self.fc2 = nn.Linear(32, 2)
        self.lif_out = snn.Leaky(beta=beta, threshold=thresh)

    def forward(self, x):
        """x: (batch, 3, H, W) — a single static image batch, NOT pre-encoded into spikes."""
        x = self.tsgf(x)  # preprocessing happens once, same as the baseline CNN

        # membrane potentials all start at zero
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_sum = 0  # accumulate output spikes across all time steps

        for t in range(NUM_STEPS):
            # the SAME preprocessed image is "shown" every time step — each
            # LIF layer's own accumulate/leak/fire dynamics is what produces
            # a genuinely time-varying spike pattern from this constant input
            cur1 = self.pool(self.conv1(x))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.pool(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            flat = self.gap(spk3).flatten(1)
            cur4 = self.fc1(flat)
            spk4, mem4 = self.lif4(cur4, mem4)

            cur_out = self.fc2(spk4)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)

            spk_out_sum = spk_out_sum + spk_out  # tally how many times each output neuron fired

        return spk_out_sum  # shape (batch, 2) — spike COUNT per class over 25 steps


def create_healthy_model(probe_batch: torch.Tensor, max_attempts: int = 10) -> "SpikingCNN":
    """
    Spiking networks can be "born dead" from an unlucky random weight
    initialization — every layer's current happens to sit below threshold,
    so nothing ever fires anywhere, for any input. A dead network also can't
    train its way out of this (no spikes -> no gradient signal). So: create
    a model, run one real batch through it as a smoke test, and if literally
    nothing fired anywhere, throw it away and try a fresh random init.
    """
    for attempt in range(1, max_attempts + 1):
        model = SpikingCNN()
        with torch.no_grad():
            out = model(probe_batch)
        total_spikes = out.sum().item()
        if total_spikes > 0:
            print(f"[create_healthy_model] attempt {attempt}: alive ({total_spikes:.0f} output spikes) — using this init")
            return model
        print(f"[create_healthy_model] attempt {attempt}: dead (0 output spikes) — retrying with new init")
    raise RuntimeError(f"Could not find a non-dead initialization in {max_attempts} attempts")
