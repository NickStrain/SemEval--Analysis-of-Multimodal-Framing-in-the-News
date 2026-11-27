# Session s6: project u = unit(mean h_fail - mean h_ok) at hidden layer 12 out of
# encoder.layer.11 attention-output and FFN-output weights, a = 0.05.
import numpy as np

FAIL = ['EN_M_TRAIN_14#0', 'EN_M_TRAIN_34#5', 'EN_M_TRAIN_5#27', 'EN_M_TRAIN_66#1', 'EN_M_TRAIN_66#7', 'EN_M_TRAIN_66#8', 'EN_M_TRAIN_79#17', 'EN_M_TRAIN_79#2', 'EN_TRAIL_3#0', 'EN_TRAIL_3#1', 'EN_TRAIL_3#17', 'EN_TRAIL_3#2', 'EN_TRAIL_3#21', 'EN_TRAIL_3#22', 'EN_TRAIL_3#23', 'EN_TRIAL_5#10', 'EN_TRIAL_5#3', 'EN_TRIAL_5#4', 'EN_TRIAL_5#5', 'EN_TRIAL_5#9']
OK = ['EN_M_TRAIN_66#3', 'EN_M_TRAIN_70#5', 'EN_M_TRAIN_79#7', 'EN_M_TRAIN_85#7', 'EN_M_TRAIN_92#0', 'EN_M_TRAIN_92#1', 'EN_M_TRAIN_92#2', 'EN_TRAIL_3#13', 'EN_TRAIL_3#24', 'EN_TRIAL_5#7']


def edit(P, H):
    z = H.load_read("read1")
    ids = list(z["ids"]); Hs = z["L12"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.11.attention.output.dense.weight", "enc.encoder.layer.11.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.05)
