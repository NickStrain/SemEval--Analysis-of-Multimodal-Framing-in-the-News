# Allowed: shift the same-named bias in every layer by a constant.
def edit(P, H):
    for l in range(12):
        P[f"enc.encoder.layer.{l}.output.dense.bias"] += 0.01
