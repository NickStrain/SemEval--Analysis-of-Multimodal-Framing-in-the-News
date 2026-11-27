# Session s2: shift intermediate.dense.bias (pre-GELU) in every layer by the constant -0.05.
# (output.dense.bias / attention.output.dense.bias shifts are cancelled exactly by the following LayerNorm.)


def edit(P, H):
    for l in range(12):
        P[f"enc.encoder.layer.{l}.intermediate.dense.bias"] += -0.05
