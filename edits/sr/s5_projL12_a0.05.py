# Session s5: project u = unit(mean h_fail - mean h_ok) at hidden layer 12 out of
# encoder.layer.11 attention-output and FFN-output weights, a = 0.05.
import numpy as np

FAIL = ['EN_M_TRAIN_11#0', 'EN_M_TRAIN_11#1', 'EN_M_TRAIN_11#2', 'EN_M_TRAIN_18#2', 'EN_M_TRAIN_25#7', 'EN_M_TRAIN_26#3', 'EN_M_TRAIN_3#6', 'EN_M_TRAIN_36#4', 'EN_M_TRAIN_39#0', 'EN_M_TRAIN_39#7', 'EN_M_TRAIN_45#1', 'EN_M_TRAIN_45#4', 'EN_M_TRAIN_5#16', 'EN_M_TRAIN_54#5', 'EN_M_TRAIN_55#3', 'EN_M_TRAIN_79#12', 'EN_M_TRAIN_79#3', 'EN_M_TRAIN_79#7', 'EN_M_TRAIN_89#7', 'EN_TRIAL_5#4']
OK = ['EN_M_TRAIN_11#9', 'EN_M_TRAIN_14#2', 'EN_M_TRAIN_29#3', 'EN_M_TRAIN_34#2', 'EN_M_TRAIN_36#13', 'EN_M_TRAIN_36#14', 'EN_M_TRAIN_46#0', 'EN_M_TRAIN_46#4', 'EN_M_TRAIN_47#11', 'EN_M_TRAIN_47#7', 'EN_M_TRAIN_54#0', 'EN_M_TRAIN_54#7', 'EN_M_TRAIN_55#0', 'EN_M_TRAIN_77#0', 'EN_M_TRAIN_77#3', 'EN_M_TRAIN_82#0', 'EN_TRAIL_3#4', 'EN_TRAIL_3#9', 'EN_TRIAL_5#6', 'EN_TRIAL_5#7']


def edit(P, H):
    z = H.load_read("read1")
    ids = list(z["ids"]); Hs = z["L12"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.11.attention.output.dense.weight", "enc.encoder.layer.11.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.05)
