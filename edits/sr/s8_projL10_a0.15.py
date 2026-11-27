# Session s8: project u = unit(mean h_fail - mean h_ok) at hidden layer 10 out of
# encoder.layer.9 attention-output and FFN-output weights, a = 0.15.
import numpy as np

FAIL = ['EN_M_TRAIN_11#0', 'EN_M_TRAIN_39#7', 'EN_M_TRAIN_45#0', 'EN_M_TRAIN_45#6', 'EN_M_TRAIN_5#0', 'EN_M_TRAIN_5#4', 'EN_M_TRAIN_5#7', 'EN_M_TRAIN_54#2', 'EN_M_TRAIN_79#19', 'EN_M_TRAIN_82#1', 'EN_M_TRAIN_92#2', 'EN_TRAIL_3#0', 'EN_TRAIL_3#12', 'EN_TRAIL_3#13', 'EN_TRAIL_3#15', 'EN_TRAIL_3#21', 'EN_TRAIL_3#22', 'EN_TRAIL_3#23', 'EN_TRAIL_3#5', 'EN_TRAIL_3#6']
OK = ['EN_M_TRAIN_25#1', 'EN_M_TRAIN_25#3', 'EN_M_TRAIN_25#5', 'EN_M_TRAIN_45#2', 'EN_M_TRAIN_5#15', 'EN_M_TRAIN_5#5', 'EN_M_TRAIN_54#1', 'EN_M_TRAIN_66#10', 'EN_M_TRAIN_70#4', 'EN_M_TRAIN_77#3', 'EN_M_TRAIN_77#7', 'EN_M_TRAIN_77#9', 'EN_M_TRAIN_79#1', 'EN_M_TRAIN_79#11', 'EN_M_TRAIN_79#7', 'EN_TRAIL_3#11', 'EN_TRAIL_3#19', 'EN_TRAIL_3#20', 'EN_TRAIL_3#7', 'EN_TRAIL_3#9']


def edit(P, H):
    z = H.load_read("read1")
    ids = list(z["ids"]); Hs = z["L10"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.9.attention.output.dense.weight", "enc.encoder.layer.9.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.15)
