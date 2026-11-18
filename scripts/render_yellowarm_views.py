"""Render YellowArm (URDF meshes, zero-joint pose) from 4 views for demo_giam_khao/0_thiet_ke_YellowArm. Needs matplotlib+trimesh:
  set PYTHONPATH=..\YellowArmSim\cad_tools && env\python.exe scripts\render_yellowarm_views.py OUT_DIR
Output is uncropped (white margin); the demo copies were cropped to the arm."""
import sys, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np, trimesh, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from mhh_paths import YELLOWARM_DIR as SRC  # noqa: E402
OUT = Path(sys.argv[1])

def origin(node):
    if node is None:
        return np.eye(4)
    t = trimesh.transformations.euler_matrix(*np.fromstring(node.get("rpy", "0 0 0"), sep=" "), axes="sxyz")
    t[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ") * 1000
    return t

robot = ET.parse(SRC / "robots/yellow_arm.urdf").getroot()
world, pending = {"yellowarm_link0": np.eye(4)}, list(robot.findall("joint"))
while pending:
    for j in pending[:]:
        p = j.find("parent").get("link")
        if p in world:
            world[j.find("child").get("link")] = world[p] @ origin(j.find("origin")); pending.remove(j)
meshes = []
for link in robot.findall("link"):
    v = link.find("visual")
    m = trimesh.load_mesh(SRC / "meshes/visual" / Path(v.find("geometry/mesh").get("filename")).name)
    m.apply_scale(1000); m.apply_transform(world[link.get("name")] @ origin(v.find("origin")))
    meshes.append(m)
base = ["#697782", "#efbd24", "#82949b", "#efbd24", "#82949b", "#efbd24", "#82949b", "#537269", "#efbd24", "#efbd24"]
allv = np.vstack([m.vertices for m in meshes]); lo, hi = allv.min(0), allv.max(0)
c, span = (lo + hi) / 2, (hi - lo).max() * 0.52

def render(name, elev, azim, title):
    fig = plt.figure(figsize=(10, 7.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    e, a = np.radians(elev), np.radians(azim)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    light = view + np.array([0.3, -0.3, 0.6]); light /= np.linalg.norm(light)
    tris, cols, depth = [], [], []
    for m, col in zip(meshes, base):
        rgb = np.array(matplotlib.colors.to_rgb(col))
        shade = 0.45 + 0.55 * np.clip(m.face_normals @ light, 0, 1)
        tris.append(m.triangles); cols.append(np.clip(rgb[None] * shade[:, None], 0, 1))
        depth.append(m.triangles.mean(1) @ view)
    tris, cols, depth = np.concatenate(tris), np.concatenate(cols), np.concatenate(depth)
    o = np.argsort(depth)  # painter's order: far first
    ax.add_collection3d(Poly3DCollection(tris[o], facecolors=cols[o], edgecolors=cols[o] * 0.8, linewidths=0.05))
    for i, f in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        f(c[i] - span, c[i] + span)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=elev, azim=azim); ax.set_axis_off()
    pass
    fig.subplots_adjust(0, 0, 1, 0.95)
    fig.savefig(OUT / name, dpi=400, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(name)

render("3_YellowArm_goc_nghieng.png", 25, -50, "YellowArm - góc nghiêng (7 khớp + gripper, tư thế 0)")
render("4_YellowArm_nhin_ngang.png", 0, -90, "YellowArm - nhìn ngang")
render("5_YellowArm_nhin_tren.png", 89, -90, "YellowArm - nhìn từ trên")
render("6_YellowArm_phia_sau.png", 20, 140, "YellowArm - góc phía sau")
