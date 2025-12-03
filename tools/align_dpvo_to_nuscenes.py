import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_tum(path, quat_format="xyzw"):
    """
    TUM: time tx ty tz qx qy qz qw
    quat_format: 'xyzw' or 'wxyz' describing how the 4 numbers are stored in file.
    Returns ts(N,), pos(N,3), quat(N,4) as [x,y,z,w].
    """
    ts, pos, quat = [], [], []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s[0] in "#;%/":
                continue
            p = s.split()
            if len(p) < 8:
                continue
            t, tx, ty, tz = map(float, p[:4])
            q1, q2, q3, q4 = map(float, p[4:8])
            if quat_format == "xyzw":
                qx, qy, qz, qw = q1, q2, q3, q4
            else:  # wxyz
                qw, qx, qy, qz = q1, q2, q3, q4
            ts.append(t); pos.append([tx, ty, tz]); quat.append([qx, qy, qz, qw])
    return np.asarray(ts), np.asarray(pos), np.asarray(quat)

def save_tum(path, ts, pos, quat_xyzw):
    with open(path, "w") as f:
        for t, (x,y,z), (qx,qy,qz,qw) in zip(ts, pos, quat_xyzw):
            f.write(f"{t:.9f} {x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

def invert_w2c_to_c2w(t_wc, q_wc_xyzw):
    """DPVO 常见输出为世界->相机，取逆得到相机->世界"""
    R_wc = R.from_quat(q_wc_xyzw).as_matrix()
    R_cw = np.transpose(R_wc, (0,2,1))
    C_w  = (-R_cw @ t_wc[...,None]).squeeze(-1)
    q_cw = R.from_matrix(R_cw).as_quat()
    return C_w, q_cw

def nn_match(ts_a, ts_b, max_diff=0.05):
    j = 0; pairs = []
    for i, ta in enumerate(ts_a):
        while j+1 < len(ts_b) and abs(ts_b[j+1]-ta) <= abs(ts_b[j]-ta):
            j += 1
        if abs(ts_b[j]-ta) <= max_diff:
            pairs.append((i, j))
    ai = np.array([i for i,_ in pairs], dtype=int)
    bj = np.array([j for _,j in pairs], dtype=int)
    return ai, bj

def umeyama_sim3(X, Y, with_scale=True):
    """Find s,R,t s.t. Y ≈ s R X + t (X->Y)"""
    muX, muY = X.mean(0), Y.mean(0)
    X0, Y0 = X - muX, Y - muY
    C = (Y0.T @ X0) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        Vt[-1,:] *= -1
        Rm = U @ Vt
    if with_scale:
        var = (X0**2).sum() / X.shape[0]
        s = S.sum() / max(var, 1e-12)
    else:
        s = 1.0
    t = muY - s * (Rm @ muX)
    return s, Rm, t

def apply_sim3_to_pose(pos_cw, quat_cw_xyzw, s, Rw, t):
    """
    world1 -> world2: y = s*Rw*x + t
    For camera-to-world pose T_cw: R_cw2 = Rw R_cw1, p_cw2 = s*Rw*p_cw1 + t
    """
    pos2 = (s * (pos_cw @ Rw.T)) + t
    R_cw = R.from_quat(quat_cw_xyzw).as_matrix()
    R2 = Rw @ R_cw
    q2 = R.from_matrix(R2).as_quat()
    return pos2, q2

def main():
    ap = argparse.ArgumentParser(description="Align DPVO TUM trajectory to nuScenes trajectory using Sim(3)")
    ap.add_argument("--nusc", required=True, help="nuScenes traj txt: '# timestamp tx ty tz qx qy qz qw' (camera->world)")
    ap.add_argument("--est",  required=True, help="DPVO traj txt in TUM format")
    ap.add_argument("--out",  required=True, help="output aligned est TUM")
    ap.add_argument("--nusc_quat", choices=["xyzw","wxyz"], default="xyzw")
    ap.add_argument("--est_quat",  choices=["xyzw","wxyz"], default="xyzw")
    ap.add_argument("--est_w2c", action="store_true", help="DPVO poses are world->camera; invert to camera->world")
    ap.add_argument("--max_diff", type=float, default=0.05, help="timestamp matching tolerance (s)")
    ap.add_argument("--no_scale", action="store_true", help="SE3 only (no scale)")
    ap.add_argument("--by_index", action="store_true", help="associate by index instead of timestamp")
    ap.add_argument("--index_offset", type=int, default=0, help="est index offset: est[i+off] <-> gt[i]")
    args = ap.parse_args()

    # 1) load
    ts_g, Pg, Qg = load_tum(args.nusc, quat_format=args.nusc_quat)   # c->w
    ts_e, Pe, Qe = load_tum(args.est,  quat_format=args.est_quat)    # maybe w->c

    if args.est_w2c:
        Pe, Qe = invert_w2c_to_c2w(Pe, Qe)

    if args.by_index:
        N = min(len(ts_e) - max(0, args.index_offset), len(ts_g))
        assert N > 10, "too few frames after index association"
        if args.index_offset >= 0:
            ai = np.arange(args.index_offset, args.index_offset + N, dtype=int)
            bj = np.arange(0, N, dtype=int)
        else:
            ai = np.arange(0, N, dtype=int)
            bj = np.arange(-args.index_offset, -args.index_offset + N, dtype=int)
    else:
        # 2) time associate
        ai, bj = nn_match(ts_e, ts_g, args.max_diff)
        if ai.size < 10:
            raise SystemExit(f"匹配帧过少: {ai.size} (请增大 --max_diff 或改用 --by_index)")
    E = Pe[ai]; G = Pg[bj]

    # 3) Sim(3) X->Y = est->gt
    s, Rw, t = umeyama_sim3(E, G, with_scale=not args.no_scale)
    print(f"Estimated Sim(3): scale={s:.6f}, R=\n{Rw}, t={t}")  

    # 4) apply to full estimated poses (保持时间序列不变)
    Pe_aligned, Qe_aligned = apply_sim3_to_pose(Pe, Qe, s, Rw, t)

    # 5) save
    save_tum(args.out, ts_e, Pe_aligned, Qe_aligned)

    # 6) report APE on matched subset
    E2 = (s * (E @ Rw.T)) + t
    rmse = float(np.sqrt(np.mean(np.sum((E2 - G)**2, axis=1))))
    path_len = float(np.linalg.norm(np.diff(G, axis=0), axis=1).sum())
    print(f"matches: {ai.size}, scale s = {s:.6f}")
    print(f"APE RMSE (after align): {rmse:.3f} m,  path length: {path_len:.1f} m,  rel: {rmse/max(path_len,1e-6)*100:.2f}%")
    print(f"saved: {args.out}")

if __name__ == "__main__":
    main()



"""
python tools/align_dpvo_to_nuscenes.py \
  --nusc /mnt/cfs-baidu/algorithm/mengmeng01.liu/code/DPVO/output/scene-0103/trajectory.txt --nusc_quat xyzw \
  --est  /mnt/cfs-baidu/algorithm/mengmeng01.liu/code/DPVO/output/scene-0103/saved_trajectories/result.txt --est_quat xyzw \
  --out  output/scene-0103/dpvo_aligned.txt \
  --by_index
"""