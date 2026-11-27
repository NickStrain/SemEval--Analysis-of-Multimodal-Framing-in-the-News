# Session s1: project u = unit(mean h_fail - mean h_ok) at hidden layer 12 out of
# encoder.layer.11 attention-output and FFN-output weights, a = 0.05.
import numpy as np

FAIL = ['EN_M_TRAIN_14#0', 'EN_M_TRAIN_18#4', 'EN_M_TRAIN_29#4', 'EN_M_TRAIN_3#0', 'EN_M_TRAIN_3#12', 'EN_M_TRAIN_3#21', 'EN_M_TRAIN_41#0', 'EN_M_TRAIN_45#7', 'EN_M_TRAIN_5#14', 'EN_M_TRAIN_5#21', 'EN_M_TRAIN_55#2', 'EN_M_TRAIN_66#4', 'EN_M_TRAIN_66#5', 'EN_M_TRAIN_66#7', 'EN_M_TRAIN_89#0', 'EN_M_TRAIN_92#5', 'EN_TRAIL_3#12', 'EN_TRAIL_3#15', 'EN_TRAIL_3#23', 'EN_TRIAL_5#10']
OK = ['EN_M_TRAIN_11#7', 'EN_M_TRAIN_18#1', 'EN_M_TRAIN_26#5', 'EN_M_TRAIN_29#6', 'EN_M_TRAIN_3#22', 'EN_M_TRAIN_34#3', 'EN_M_TRAIN_39#5', 'EN_M_TRAIN_39#6', 'EN_M_TRAIN_47#11', 'EN_M_TRAIN_47#7', 'EN_M_TRAIN_5#5', 'EN_M_TRAIN_77#3', 'EN_M_TRAIN_79#18', 'EN_M_TRAIN_89#5', 'EN_M_TRAIN_89#6', 'EN_TRAIL_3#9', 'EN_TRIAL_5#6', 'EN_TRIAL_5#7']


def edit(P, H):
    z = H.load_read("read1")
    ids = list(z["ids"]); Hs = z["L12"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.11.attention.output.dense.weight", "enc.encoder.layer.11.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.05)
