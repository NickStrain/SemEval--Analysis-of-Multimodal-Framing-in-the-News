# Session s7: project u = unit(mean h_fail - mean h_ok) at hidden layer 10 out of
# encoder.layer.9 attention-output and FFN-output weights, a = 0.15.
import numpy as np

FAIL = ['EN_M_TRAIN_14#1', 'EN_M_TRAIN_14#3', 'EN_M_TRAIN_25#3', 'EN_M_TRAIN_25#4', 'EN_M_TRAIN_5#22', 'EN_M_TRAIN_5#25', 'EN_M_TRAIN_66#10', 'EN_M_TRAIN_66#8', 'EN_M_TRAIN_68#1', 'EN_M_TRAIN_79#10', 'EN_M_TRAIN_79#6', 'EN_M_TRAIN_79#9', 'EN_M_TRAIN_85#0', 'EN_M_TRAIN_85#11', 'EN_M_TRAIN_85#4', 'EN_M_TRAIN_92#8', 'EN_TRIAL_5#3', 'EN_TRIAL_5#4', 'EN_TRIAL_5#5', 'EN_TRIAL_5#9']
OK = ['EN_M_TRAIN_11#7', 'EN_M_TRAIN_14#2', 'EN_M_TRAIN_29#0', 'EN_M_TRAIN_29#3', 'EN_M_TRAIN_3#22', 'EN_M_TRAIN_34#3', 'EN_M_TRAIN_36#13', 'EN_M_TRAIN_39#5', 'EN_M_TRAIN_39#6', 'EN_M_TRAIN_46#0', 'EN_M_TRAIN_47#0', 'EN_M_TRAIN_47#7', 'EN_M_TRAIN_54#0', 'EN_M_TRAIN_54#7', 'EN_M_TRAIN_55#0', 'EN_M_TRAIN_77#3', 'EN_M_TRAIN_82#0', 'EN_M_TRAIN_89#6', 'EN_TRAIL_3#18', 'EN_TRAIL_3#4']


def edit(P, H):
    z = H.load_read("read1")
    ids = list(z["ids"]); Hs = z["L10"]
    u = Hs[[ids.index(i) for i in FAIL]].mean(0) - Hs[[ids.index(i) for i in OK]].mean(0)
    u = H.unit(u)
    for name in ("enc.encoder.layer.9.attention.output.dense.weight", "enc.encoder.layer.9.output.dense.weight"):
        P[name] = H.project_out(P[name], u, 0.15)
