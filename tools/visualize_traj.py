import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def load_tum(path):
    ts, txyz, qxyzw = [], [], []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s[0] in "#;%/":
                continue
            parts = s.split()
            if len(parts) < 8:
                continue
            t, tx, ty, tz, qx, qy, qz, qw = map(float, parts[:8])
            ts.append(t)
            txyz.append([tx, ty, tz])
            qxyzw.append([qx, qy, qz, qw])
    ts = np.asarray(ts, dtype=float)
    txyz = np.asarray(txyz, dtype=float)
    qxyzw = np.asarray(qxyzw, dtype=float)  # [x,y,z,w]
    return ts, txyz, qxyzw

def w2c_to_c2w(txyz, qxyzw):
    # DPVO 常用格式：T_wc（世界->相机），需要取逆得到相机在世界下的轨迹
    R_wc = R.from_quat(qxyzw).as_matrix()           # (N,3,3)
    R_cw = np.transpose(R_wc, (0, 2, 1))            # 逆
    C_w = (-R_cw @ txyz[..., None]).squeeze(-1)     # 相机位置
    q_cw_xyzw = R.from_matrix(R_cw).as_quat()
    return C_w, q_cw_xyzw

def set_axes_equal(ax):
    xs, ys, zs = [getattr(ax, f'get_{a}lim')() for a in ('x','y','z')]
    xmid, ymid, zmid = [(l[0]+l[1])/2 for l in (xs,ys,zs)]
    radius = max((xs[1]-xs[0]), (ys[1]-ys[0]), (zs[1]-zs[0]))/2
    ax.set_xlim(xmid-radius, xmid+radius)
    ax.set_ylim(ymid-radius, ymid+radius)
    ax.set_zlim(zmid-radius, zmid+radius)

def _plot_2d_plane(pos, plane, outdir, title):
    """Save 2D projection figure for plane in {'xy','xz','yz'}."""
    i, j = {"xy": (0,1), "xz": (0,2), "yz": (1,2)}[plane]
    fig, ax2 = plt.subplots(figsize=(8,7))
    ax2.plot(pos[:,i], pos[:,j], 'b-', lw=2)
    ax2.scatter(pos[0,i], pos[0,j], c='g', s=50, label='start')
    ax2.scatter(pos[-1,i], pos[-1,j], c='r', s=50, label='end')
    ax2.set_xlabel(f"{plane[0].upper()} [m]")
    ax2.set_ylabel(f"{plane[1].upper()} [m]")
    ax2.set_aspect("equal", adjustable="box")
    ax2.grid(True, ls="--", alpha=0.3)
    ax2.set_title(f"{title} (top-down: {plane})")
    fig.tight_layout()
    fig.savefig(outdir / f"traj_{plane}.png", dpi=200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description="Visualize TUM trajectory")
    ap.add_argument("--traj", required=True, help="TUM file: time tx ty tz qx qy qz qw")
    ap.add_argument("--outdir", default="traj_vis", help="output directory")
    ap.add_argument("--title", default=None)
    ap.add_argument("--w2c", action="store_true", help="输入为世界->相机(T_wc)，需取逆为相机->世界")
    ap.add_argument("--every", type=int, default=1, help="可视化下采样步长")
    args = ap.parse_args()

    ts, txyz, qxyzw = load_tum(args.traj)
    if ts.size == 0:
        raise SystemExit("空轨迹文件或解析失败")

    if args.w2c:
        pos, quat = w2c_to_c2w(txyz, qxyzw)
    else:
        pos, quat = txyz, qxyzw

    pos = pos[::args.every]
    ts = ts[::args.every]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 3D 轨迹
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(pos[:,0], pos[:,1], pos[:,2], 'b-', lw=2, label="trajectory")
    ax.scatter(pos[0,0], pos[0,1], pos[0,2], c='g', s=60, label="start")
    ax.scatter(pos[-1,0], pos[-1,1], pos[-1,2], c='r', s=60, label="end")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ttl = args.title or Path(args.traj).name
    ax.set_title(ttl); ax.legend(); set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(outdir / "traj_3d.png", dpi=200)
    plt.close(fig)

    # 新增：同时保存 xy / xz / yz 三个平面投影
    for plane in ("xy", "xz", "yz"):
        _plot_2d_plane(pos, plane, outdir, ttl)

    # 简要信息
    seg = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    print(f"poses: {len(pos)}, length: {seg.sum():.1f} m, saved to: {outdir}")

if __name__ == "__main__":
    main()