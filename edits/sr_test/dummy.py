# Harness test: project a RANDOM direction out of layer 6 attention-output and FFN-output, a = 0.05.
import torch

def edit(P, H):
    u = H.unit(torch.randn(384, generator=torch.Generator().manual_seed(0)))
    for name in ("enc.encoder.layer.6.attention.output.dense.weight", "enc.encoder.layer.6.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.05)
