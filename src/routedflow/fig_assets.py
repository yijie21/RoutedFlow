"""Generate thumbnail assets for the pipeline figure from the extracted labels.

Reads data/c_labels/<suite>/<TASK>.h5 (demo_0) and writes small illustration
PNGs into doc/fig_assets/ (rgb, contact heatmap, depth, robot/object masks,
robot/object flow, C-label pose axes, plus *_crop variants). Deterministic:
re-run any time the labels or the chosen demo change. Run via `run_fig.py assets`.
"""
import json
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image, ImageDraw

TASK = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo"
H5 = f"/workspace/code/RoutedFlow/data/c_labels/libero_spatial/{TASK}.h5"
REPO = __file__.rsplit("/src/", 1)[0]
OUT = REPO + "/doc/fig_assets"
os.makedirs(OUT, exist_ok=True)

GREEN = (46, 160, 90)
ORANGE = (222, 130, 40)


def save(name, arr):
    Image.fromarray(arr.astype(np.uint8)).save(os.path.join(OUT, name))
    print("wrote", name)


def project(w2p, pts):
    homo = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    uvw = (w2p @ homo.T).T
    return np.stack([uvw[:, 1] / uvw[:, 2], uvw[:, 0] / uvw[:, 2]], axis=1)  # row, col


def arrow_path(img, path_rc, color, width=6):
    """Polyline + arrowhead on a PIL image; path (N,2) row/col."""
    draw = ImageDraw.Draw(img)
    pts = [(c, r) for r, c in path_rc]
    draw.line(pts, fill=color, width=width, joint="curve")
    # arrowhead
    (r1, c1), (r0, c0) = path_rc[-1], path_rc[-4 if len(path_rc) > 4 else 0]
    v = np.array([c1 - c0, r1 - r0], float)
    v /= (np.linalg.norm(v) + 1e-9)
    n = np.array([-v[1], v[0]])
    tip = np.array([c1, r1]) + v * 10
    a = tip - v * 26 + n * 13
    b = tip - v * 26 - n * 13
    draw.polygon([tuple(tip), tuple(a), tuple(b)], fill=color)


with h5py.File(H5) as f:
    g = f["demo_0"]
    rgb = np.array(g["rgb0"]); depth = np.array(g["depth0"]); seg = np.array(g["seg0"])
    w2p = np.array(f["world_to_pix"])
    geom_map = json.loads(f["geom_id_to_body"][()])
    robot_gids = set(np.array(f["robot_geom_ids"]).tolist())
    ee = np.array(g["ee_pos"]); ee_quat = np.array(g["ee_quat"]); phase = np.array(g["phase"])
    obj_pos = np.array(g["obj_pos"]); obj_names = json.loads(f["obj_names"][()])
    row, col = np.array(g["contact_rowcol"])
    tg = int(g.attrs["t_g"])
    grasped = "akita_black_bowl_1"

# 1. plain rgb
save("rgb.png", rgb)

# 2. contact heatmap overlay
yy, xx = np.mgrid[0:512, 0:512]
heat = np.exp(-((yy - row) ** 2 + (xx - col) ** 2) / (2 * 18 ** 2))
hm = (cm.get_cmap("hot")(heat)[..., :3] * 255)
alpha = (0.75 * heat + 0.10)[..., None]
save("heatmap.png", rgb * (1 - alpha) + hm * alpha)

# 3. depth (turbo)
d = np.clip((depth - 0.5) / (1.6 - 0.5), 0, 1)
save("depth.png", cm.get_cmap("turbo")(d)[..., :3] * 255)

# 4/5. robot & object mask overlays (tinted on dimmed grayscale)
gray = rgb.mean(-1, keepdims=True) * 0.45 + 70
body_of = {int(k): v for k, v in geom_map.items()}
robot_m = np.isin(seg, list(robot_gids))
obj_gids = [gid for gid, b in body_of.items() if b and b.startswith(grasped)]
obj_m = np.isin(seg, obj_gids)
for name, m, c in (("mask_robot.png", robot_m, (225, 70, 70)),
                   ("mask_object.png", obj_m, GREEN)):
    img = np.repeat(gray, 3, axis=-1)
    img[m] = 0.25 * img[m] + 0.75 * np.array(c)
    save(name, img)

def smooth(p, k=5):
    if len(p) < k:
        return p
    ker = np.ones(k) / k
    return np.stack([np.convolve(p[:, i], ker, mode="valid") for i in (0, 1)], axis=1)


# 6. robot flow (approach): EE path 0..tg + short offset streamlines
img = Image.fromarray(rgb.copy())
path = smooth(project(w2p, ee[: tg + 1])[::2])
for off, trim in (((-18, -10), 4), ((0, 0), 0), ((15, 11), 4)):
    p = path[: len(path) - trim] + np.array(off)
    arrow_path(img, p.tolist(), GREEN, width=4 if off != (0, 0) else 7)
d = ImageDraw.Draw(img)
d.ellipse([col - 9, row - 9, col + 9, row + 9], outline=(255, 220, 40), width=5)
save("flow_robot.png", np.array(img))

# 7. object flow (transport): grasped object path tg..end
img = Image.fromarray(rgb.copy())
gi = [i for i, n in enumerate(obj_names) if n.startswith(grasped)][0]
opath = smooth(project(w2p, obj_pos[tg:, gi])[::3], k=7)
for off, trim in (((0, 0), 0), ((17, -14), 5)):
    p = opath[: len(opath) - trim] + np.array(off)
    arrow_path(img, p.tolist(), ORANGE, width=4 if off != (0, 0) else 7)
save("flow_object.png", np.array(img))

# 8. C label illustration: star + R(tg) axes (reuse quat->mat)
from routedflow.viz_c_labels import quat_to_mat  # noqa: E402
img = Image.fromarray(rgb.copy())
d = ImageDraw.Draw(img)
Rm = quat_to_mat(ee_quat[tg])
for k, c in enumerate(((235, 60, 60), (60, 190, 60), (70, 110, 235))):
    tip_w = ee[tg] + 0.09 * Rm[:, k]
    tr, tc = project(w2p, tip_w[None])[0]
    d.line([(col, row), (tc, tr)], fill=c, width=7)
d.ellipse([col - 12, row - 12, col + 12, row + 12], fill=(255, 220, 40), outline=(0, 0, 0), width=3)
save("c_label.png", np.array(img))

# crops (focus on the action area) for tighter thumbs
for name, box in (("flow_robot.png", (60, 100, 460, 420)), ("flow_object.png", (100, 140, 500, 460)),
                  ("c_label.png", (120, 130, 420, 380)), ("heatmap.png", (120, 130, 420, 380))):
    im = Image.open(os.path.join(OUT, name)).crop(box)
    im.save(os.path.join(OUT, name.replace(".png", "_crop.png")))
print("done")
