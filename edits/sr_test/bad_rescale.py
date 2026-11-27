# Must be rejected: whole-tensor rescaling.
def edit(P, H):
    k = "enc.encoder.layer.3.output.dense.weight"; P[k] = P[k] * 0.9
