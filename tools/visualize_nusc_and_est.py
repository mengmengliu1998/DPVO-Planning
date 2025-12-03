import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_traj_txt(path, quat_format="xyzw"):
    """
    读取一类通用 txt：每行至少包含
      timestamp tx ty tz q1 q2 q3 q4
    忽略以 # // ; % 开头的注释行。
    返回: ts(N,), pos(N,3), quat(N,4) 统一为 [x,y,z,w]
    """
    ts, pos, quat = [], [], []
    with open(path, "r") as f:
        for ln in f:
            s = ln.strip()
            if not s or s[0] in "#;/%" or s.startswith("//"):
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            t, tx, ty, tz = map(float, parts[:4])
            a, b, c, d = map(float, parts[4:8])
            if quat_format == "xyzw":
                qx, qy, qz, qw = a, b, c, d
            else:  # wxyz
                qw, qx, qy, qz = a, b, c, d
            ts.append(t); pos.append([tx, ty, tz]); quat.append([qx, qy, qz, qw])
    ts = np.asarray(ts, dtype=float)
    pos = np.asarray(pos, dtype=float)
    quat = np.asarray(quat, dtype=float)
    # 按时间排序（以防输入无序）
    if ts.size > 0:
        order = np.argsort(ts)
        ts, pos, quat = ts[order], pos[order], quat[order]
    return ts, pos, quat

def invert_w2c_to_c2w(t_wc, q_wc_xyzw):
    """如果估计轨迹是世界->相机，把它变成相机->世界。"""
    R_wc = R.from_quat(q_wc_xyzw).as_matrix()
    R_cw = np.transpose(R_wc, (0, 2, 1))
    C_w = (-R_cw @ t_wc[..., None]).squeeze(-1)
    q_cw = R.from_matrix(R_cw).as_quat()
    return C_w, q_cw

def set_axes_equal(ax):
    """让 3D 轴等比例显示"""
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xmid = np.mean(xlim); ymid = np.mean(ylim); zmid = np.mean(zlim)
    r = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2.0
    ax.set_xlim3d([xmid-r, xmid+r])
    ax.set_ylim3d([ymid-r, ymid+r])
    ax.set_zlim3d([zmid-r, zmid+r])

def plot_plane(pos_gt, pos_est, plane, outdir, title, colors=("tab:blue","tab:red")):
    idx = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}[plane]
    i, j = idx
    fig, ax = plt.subplots(figsize=(8,7))
    ax.plot(pos_gt[:,i],  pos_gt[:,j],  color=colors[0], lw=2, label="nuScenes (GT)")
    ax.plot(pos_est[:,i], pos_est[:,j], color=colors[1], lw=2, label="DPVO (est)")
    ax.scatter(pos_gt[0,i],  pos_gt[0,j],  c=colors[0], s=60, marker="o")
    ax.scatter(pos_gt[-1,i], pos_gt[-1,j], c=colors[0], s=60, marker="s")
    ax.scatter(pos_est[0,i],  pos_est[0,j],  c=colors[1], s=60, marker="o")
    ax.scatter(pos_est[-1,i], pos_est[-1,j], c=colors[1], s=60, marker="s")
    ax.set_xlabel(f"{plane[0].upper()} [m]")
    ax.set_ylabel(f"{plane[1].upper()} [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()
    ax.set_title(f"{title} (plane: {plane})")
    fig.tight_layout()
    fig.savefig(outdir / f"overlay_{plane}.png", dpi=220)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser("Overlay visualize nuScenes traj and DPVO traj")
    ap.add_argument("--nusc", required=True, help="nuScenes格式：timestamp tx ty tz qx qy qz qw (camera->world)")
    ap.add_argument("--est",  required=True, help="DPVO TUM格式：timestamp tx ty tz qx qy qz qw")
    ap.add_argument("--outdir", default="traj_overlay_vis")
    ap.add_argument("--title",  default=None)
    ap.add_argument("--nusc_quat", choices=["xyzw","wxyz"], default="xyzw",
                    help="nuScenes文件中的四元数顺序，默认 xyzw")
    ap.add_argument("--est_quat",  choices=["xyzw","wxyz"], default="xyzw",
                    help="DPVO文件中的四元数顺序，默认 xyzw")
    ap.add_argument("--est_w2c", action="store_true",
                    help="若 DPVO 是世界->相机，则勾选以转为相机->世界再可视化")
    ap.add_argument("--every", type=int, default=1, help="下采样步长，可视化提速")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 读取
    ts_g, Pg, Qg = load_traj_txt(args.nusc, quat_format=args.nusc_quat)      # c->w
    ts_e, Pe, Qe = load_traj_txt(args.est,  quat_format=args.est_quat)       # ?->?
    if args.est_w2c:
        Pe, Qe = invert_w2c_to_c2w(Pe, Qe)

    if Pg.size == 0 or Pe.size == 0:
        raise SystemExit("读取失败：输入轨迹为空")

    # 下采样（仅可视化）
    Pg_v = Pg[::args.every]; Pe_v = Pe[::args.every]

    # 3D 叠加
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Pg_v[:,0], Pg_v[:,1], Pg_v[:,2], color="tab:blue", lw=2, label="nuScenes (GT)")
    ax.plot(Pe_v[:,0], Pe_v[:,1], Pe_v[:,2], color="tab:red",  lw=2, label="DPVO (est)")
    ax.scatter(Pg_v[0,0], Pg_v[0,1], Pg_v[0,2], c="tab:blue", s=60, marker="o")
    ax.scatter(Pg_v[-1,0], Pg_v[-1,1], Pg_v[-1,2], c="tab:blue", s=60, marker="s")
    ax.scatter(Pe_v[0,0], Pe_v[0,1], Pe_v[0,2], c="tab:red",  s=60, marker="o")
    ax.scatter(Pe_v[-1,0], Pe_v[-1,1], Pe_v[-1,2], c="tab:red",  s=60, marker="s")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ttl = args.title or f"{Path(args.nusc).stem} vs {Path(args.est).stem}"
    ax.set_title(ttl); ax.legend(); set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(outdir / "overlay_3d.png", dpi=220)
    plt.close(fig)

    # 平面投影：xy / xz / yz
    for plane in ("xy", "xz", "yz"):
        plot_plane(Pg_v, Pe_v, plane, outdir, ttl)

    # 简要统计
    Lg = float(np.linalg.norm(np.diff(Pg, axis=0), axis=1).sum())
    Le = float(np.linalg.norm(np.diff(Pe, axis=0), axis=1).sum())
    print(f"nuScenes poses={len(Pg)} length={Lg:.1f} m | DPVO poses={len(Pe)} length={Le:.1f} m")
    print(f"saved to: {outdir.resolve()}")

if __name__ == "__main__":
    main()


"""
python tools/visualize_nusc_and_est.py \
  --nusc /mnt/cfs-baidu/algorithm/mengmeng01.liu/code/DPVO/output/scene-0103/trajectory.txt \
  --est  /mnt/cfs-baidu/algorithm/mengmeng01.liu/code/DPVO/output/scene-0103/dpvo_aligned.txt \
  --outdir output/scene-0103/overlay_vis
"""