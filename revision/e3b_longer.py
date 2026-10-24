# Van Van Chau(1), Hao Anh Nguyen(1), Tuan Minh Nguyen(1,*), Chinda Thorn(2), and Phayung Meesad(2,*)
# (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
#     Ho Chi Minh City, Viet Nam, vancv@ptit.edu.vn, haona@ptit.edu.vn, minhtuan@ptit.edu.vn
# (2) Department of Information Technology Management, King Mongkut's University of Technology North Bangkok,
#     Bangkok, Thailand, s6807011866029@email.kmutnb.ac.th, phayung.m@itd.kmutnb.ac.th
# (*) Corresponding authors
#
"""E3b diagnostic only (not a reported configuration): does QCAE-manual leave the grey initial state with more epochs?"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import run_revision as r
torch.set_num_threads(2)  # same thread count as all other revision runs
x_train, x_val, x_test = r.data(); tv = np.concatenate([x_train, x_val])
hp = {"lr": 1e-3, "weight_decay": 0.0, "p_layers": 2, "hidden_channels": 8}
out = {}
for ep in [80, 200, 400]:
    r.FINAL_EPOCHS = ep
    res, m = r.retrain_and_eval("qcae", hp, tv, x_test, 7)
    out[ep] = res
    print(ep, round(res["mean_ssim"], 4), round(res["final_train_loss"], 4), flush=True)
json.dump(out, open("results/E3b_manual_longer.json", "w"), indent=2)
