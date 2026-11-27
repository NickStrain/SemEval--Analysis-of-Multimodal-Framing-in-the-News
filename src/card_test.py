# Test decision rules suggested by the diagnosis, all nested (chosen on 4 folds, applied to the 5th).
import itertools, warnings, numpy as np, lightgbm as lgb
warnings.filterwarnings("ignore")
from src.data import load_cv
from src.en import load_en
from src.ens import weight_grid
from src.post import _micro
from src.step10 import FINE, load_deberta, load_train_cv
cv = load_cv(); Ya = np.stack(cv.labels).astype(bool); em = (cv.lang == "en").values
en, Y, F = load_en(); n = len(Y)
mc,_,_ = load_train_cv("phase1b/M_E1_extraEN", em, cv.fold.values, Ya, cv.lang.values)
xl,_,_ = load_train_cv("phase1c/XL_E1", em, cv.fold.values, Ya, cv.lang.values)
db,_,_ = load_deberta("deb_silver_bce", Y, F); oofs=[mc,xl,db]
# ensemble scores, weights nested
S = np.zeros(Y.shape, np.float32)
for k in range(5):
    tr,te=F!=k,F==k
    w=max(weight_grid(3,0.1), key=lambda w: max(_micro(Y[tr], sum(a*o for a,o in zip(w,oofs))[tr]>=t) for t in FINE[::2]))
    S[te]=sum(a*o[te] for a,o in zip(w,oofs))
card=Y.sum(1); order=np.argsort(-S,1); Ss=-np.sort(-S,1)
def rep(name,P):
    r=np.array([_micro(Y[F==k],P[F==k]) for k in range(5)]); print(f"{name:55s} {100*r.mean():.2f} ± {100*r.std():.2f}")
def topk(kvec):
    P=np.zeros(Y.shape,bool)
    for i in range(n): P[i,order[i,:kvec[i]]]=True
    return P
# A: global threshold (baseline)
P=np.zeros(Y.shape,bool)
for k in range(5):
    tr,te=F!=k,F==k; t=max(FINE,key=lambda t:_micro(Y[tr],S[tr]>=t)); P[te]=S[te]>=t
rep("A global threshold (current)",P)
import json
pa = json.load(open("teacher/pilot_articles.json")); pa = set(pa if isinstance(pa, list) else pa.keys())
m = en.article_id.isin(pa).values
P=np.zeros(Y.shape,bool)
for k in range(5):
    tr,te=F!=k,F==k; t=max(FINE,key=lambda t:_micro(Y[tr],S[tr]>=t)); P[te]=S[te]>=t
print("pilot paragraphs", m.sum(), "best ensemble micro on them", round(100*_micro(Y[m],P[m]),1))
