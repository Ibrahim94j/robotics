
"""
Creates a synthetic test video with known Ground Truth.

A virtual camera looks at a 2500 x 1500 mm field from a side angle.
Two cars (3D boxes with height, like real cars -> realistic perspective error)
drive for 9 seconds and then stay still for 3 seconds
(for measuring the mapping error).

Output (in the target folder):
  synthetic.mp4              Video, 1280x720 @ 60 FPS
  synthetic_truth.csv        Ground Truth for each frame and car (x, y, theta, u, v)
  synthetic_static_truth.csv Static Ground Truth in the format used by evaluate.py mapping
  calibration_floor.json     Correct calibration (same format as sam_tracking.py)

Usage:  python tools/make_synthetic_video.py --out synthetic
"""

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1280, 720, 60
FIELD = (2500.0, 1500.0)
CAR_SIZE = (200.0, 100.0, 70.0)        # Länge, Breite, Höhe in mm
CARS = [("Red Racer", (40, 40, 220)), ("Green Hornet", (60, 190, 40))]
MOVE_S, STILL_S = 9.0, 3.0


def camera():
    cam = np.array([1250.0, -1900.0, 1600.0])
    target = np.array([1250.0, 700.0, 0.0])
    z = (target - cam) / np.linalg.norm(target - cam)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    K = np.array([[950.0, 0, W / 2], [0, 950.0, H / 2], [0, 0, 1]])
    return K, R, -R @ cam


def project(points, K, R, t):
    p = (R @ np.asarray(points, float).T).T + t
    p = (K @ p.T).T
    return p[:, :2] / p[:, 2:3]


def pose(car, s):
    """Position (mm) und Fahrtrichtung (Grad) zum Zeitpunkt s."""
    s = min(s, MOVE_S)
    if car == 0:   # Ellipse
        a = 1.1 * s
        x, y = 1250 + 850 * math.cos(a), 750 + 480 * math.sin(a)
        dx, dy = -850 * math.sin(a), 480 * math.cos(a)
    else:          # liegende Acht
        x, y = 1250 + 900 * math.sin(0.7 * s), 750 + 380 * math.sin(1.4 * s)
        dx, dy = 630 * math.cos(0.7 * s), 532 * math.cos(1.4 * s)
    return x, y, math.degrees(math.atan2(dy, dx)) % 360


def box_corners(x, y, theta):
    L, B, Hc = CAR_SIZE
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    base = [(x + c * px - s * py, y + s * px + c * py)
            for px, py in [(L / 2, B / 2), (L / 2, -B / 2), (-L / 2, -B / 2), (-L / 2, B / 2)]]
    return [(bx, by, 0.0) for bx, by in base], [(bx, by, Hc) for bx, by in base]


def floor_image(K, R, t, rng):
    # Draufsicht-Textur: helle Fliesen (600 mm) mit Fugen, leichte Struktur.
    tex = np.full((750, 1250, 3), (175, 178, 182), np.float32)
    tex += cv2.GaussianBlur(rng.normal(0, 6, tex.shape).astype(np.float32), (0, 0), 3)
    tex = np.clip(tex, 0, 255).astype(np.uint8)
    for gx in range(0, 1251, 300):
        cv2.line(tex, (gx, 0), (gx, 749), (150, 152, 155), 2)
    for gy in range(0, 751, 300):
        cv2.line(tex, (0, gy), (1249, gy), (150, 152, 155), 2)
    tex_pts = np.float32([[0, 0], [1250, 0], [1250, 750], [0, 750]])
    img_pts = project([(0, 0, 0), (2500, 0, 0), (2500, 1500, 0), (0, 1500, 0)], K, R, t).astype(np.float32)
    Hm = cv2.getPerspectiveTransform(tex_pts, img_pts)
    img = np.full((H, W, 3), (95, 100, 105), np.uint8)       # Umgebung
    warped = cv2.warpPerspective(tex, Hm, (W, H))
    covered = cv2.warpPerspective(np.full((750, 1250), 255, np.uint8), Hm, (W, H))
    img[covered > 0] = warped[covered > 0]
    return img, img_pts


def draw_car(img, colour, x, y, theta, K, R, t):
    bottom, top = box_corners(x, y, theta)
    pb, pt = project(bottom, K, R, t), project(top, K, R, t)
    hull = cv2.convexHull(np.vstack([pb, pt]).astype(np.int32))
    dark = tuple(int(c * 0.6) for c in colour)
    cv2.fillConvexPoly(img, hull, dark)                          # Seitenflächen
    cv2.fillConvexPoly(img, pt.astype(np.int32), colour)          # Dach
    # Windschutzscheibe vorne (macht die Autos asymmetrisch, wie echte)
    c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
    L, B, Hc = CAR_SIZE
    wind = [(x + c * px - s * py, y + s * px + c * py, Hc)
            for px, py in [(L * 0.25, B * 0.4), (L * 0.25, -B * 0.4), (L * 0.05, -B * 0.4), (L * 0.05, B * 0.4)]]
    cv2.fillConvexPoly(img, project(wind, K, R, t).astype(np.int32), (60, 50, 40))
    return project([(x, y, Hc / 2)], K, R, t)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="synthetic")
    p.add_argument("--noise", type=float, default=4.0)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    K, R, t = camera()
    floor, corners = floor_image(K, R, t, rng)

    calib = dict(schema_version=1, image_size=[W, H], coordinate_space="original_image_pixels",
                 image_points=corners.tolist(), width_mm=FIELD[0], height_mm=FIELD[1],
                 homography=cv2.getPerspectiveTransform(
                     corners, np.float32([[0, 0], [FIELD[0], 0], [FIELD[0], FIELD[1]], [0, FIELD[1]]])).tolist(),
                 note="Synthetic calibration (exact floor corners).")
    (out / "calibration_floor.json").write_text(json.dumps(calib, indent=2), encoding="utf-8")

    writer = cv2.VideoWriter(str(out / "synthetic.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    frames = int((MOVE_S + STILL_S) * FPS)
    with open(out / "synthetic_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "timestamp_s", "car_id", "x_mm", "y_mm", "theta_deg", "u", "v"])
        for i in range(frames):
            s = i / FPS
            img = floor.copy()
            for ci, (name, colour) in enumerate(CARS):
                x, y, th = pose(ci, s)
                u, v = draw_car(img, colour, x, y, th, K, R, t)
                w.writerow([i, f"{s:.6f}", name, f"{x:.2f}", f"{y:.2f}", f"{th:.2f}", f"{u:.1f}", f"{v:.1f}"])
            noise = rng.normal(0, args.noise, img.shape)
            writer.write(np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    writer.release()

    with open(out / "synthetic_static_truth.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["car_id", "t_start_s", "t_end_s", "x_mm", "y_mm", "theta_deg"])
        for ci, (name, _) in enumerate(CARS):
            x, y, th = pose(ci, MOVE_S)
            w.writerow([name, MOVE_S + 0.5, MOVE_S + STILL_S, f"{x:.2f}", f"{y:.2f}", f"{th:.2f}"])
    print(f"Fertig: {out}/ ({frames} Frames)")


if __name__ == "__main__":
    main()