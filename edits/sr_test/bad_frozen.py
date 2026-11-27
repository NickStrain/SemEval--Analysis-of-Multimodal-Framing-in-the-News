# Must be rejected: the classifier head is frozen (not even present in P).
def edit(P, H):
    P["head.1.weight"] = P["enc.encoder.layer.0.output.dense.weight"][:15, :768] * 0
