"""
Evaluation of the live tracker for the report.

1) annotate – Create Ground Truth by clicking with the mouse (where is each car?)
       python evaluate.py annotate --video test.mp4 --cars "Red Racer,Green Hornet" --step 10 --out gt.csv

2) roc – ROC curve of the color classifier (the color similarity threshold is varied)
       python evaluate.py roc --video test.mp4 --gt gt.csv --cars "Red Racer,Green Hornet"

3) mapping – Mapping error for stationary cars at measured positions
       python live_tracker.py --source statisch.mp4 --no-udp --log statisch_log.csv
       python evaluate.py mapping --log statisch_log.csv --truth positionen.csv

   positionen.csv (one line per measurement position, times in seconds of the video):
       car_id,t_start_s,t_end_s,x_mm,y_mm,theta_deg
       Red Racer,2.0,6.0,600,300,0
       Red Racer,9.0,13.0,1800,900,90

The Ground Truth file (gt.csv) has the columns frame,car_id,u,v (original pixels,
-1,-1 = car is not visible).
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import live_tracker as lt

_trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 1.x und 2.x


# ---------------------------------------------------------------------------
# 1) Annotieren
# ---------------------------------------------------------------------------

def annotate(args):
    names = [n.strip() for n in args.cars.split(",") if n.strip()]
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {args.video}")
    rows, clicks = [], []
    window = "Annotieren"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print("Linksklick = Mitte des Autos, N = nicht sichtbar, S = Frame überspringen, "
          "Backspace = letzte Eingabe zurück, Q = speichern und beenden", flush=True)
    frame_index = -1
    quit_all = False
    while not quit_all:
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        if frame_index % args.step:
            continue
        answers = []
        while len(answers) < len(names):
            name = names[len(answers)]
            view = frame.copy()
            for (u, v), n in zip(answers, names):
                if u >= 0:
                    cv2.circle(view, (u, v), 6, (0, 255, 255), 2)
                    cv2.putText(view, n, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            lt.overlay_text(view, [(f"Frame {frame_index}: '{name}' anklicken (N = nicht sichtbar)",
                                    (255, 255, 255))])
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 0xFF
            if clicks:
                answers.append(clicks.pop())
                clicks.clear()
            elif key in (ord("n"), ord("N")):
                answers.append((-1, -1))
            elif key in (ord("s"), ord("S")):
                answers = None
                break
            elif key == 8 and answers:
                answers.pop()
            elif key in (ord("q"), ord("Q"), 27):
                quit_all = True
                answers = None
                break
        if answers:
            rows.extend([frame_index, n, u, v] for n, (u, v) in zip(names, answers))
    cap.release()
    cv2.destroyAllWindows()
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "car_id", "u", "v"])
        w.writerows(rows)
    print(f"{len(rows) // max(1, len(names))} Frames annotiert -> {args.out}")


# ---------------------------------------------------------------------------
# 2) ROC
# ---------------------------------------------------------------------------

def roc_points(scores, labels):
    """TPR/FPR für jede Schwelle (absteigend). Nicht gefundene Positive haben score = -inf."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    P, N = labels.sum(), (1 - labels).sum()
    if P == 0 or N == 0:
        raise SystemExit("Für eine ROC-Kurve braucht es positive und negative Beispiele.")
    thresholds = np.unique(scores[np.isfinite(scores)])[::-1]
    tpr = [0.0] + [float(((scores >= th) & (labels == 1)).sum() / P) for th in thresholds]
    fpr = [0.0] + [float(((scores >= th) & (labels == 0)).sum() / N) for th in thresholds]
    tpr.append(tpr[-1])
    fpr.append(1.0)
    return np.array(fpr), np.array(tpr), np.concatenate([[np.inf], thresholds, [-np.inf]])


def collect_samples(args):
    names = [n.strip() for n in args.cars.split(",") if n.strip()]
    gt = defaultdict(dict)
    with open(args.gt, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["car_id"] in names:
                gt[int(row["frame"])][row["car_id"]] = (float(row["u"]), float(row["v"]))
    cap = cv2.VideoCapture(args.video)
    ok, first = cap.read()
    if not ok:
        raise SystemExit("Video nicht lesbar.")
    work_size, scale = lt.work_size_for(first, args.work_width)
    background = cv2.imread(args.background)
    if background is None or background.shape[:2] != (work_size[1], work_size[0]):
        background = lt.background_from_video(args.video, work_size, args.bg_seconds)
    detector = lt.Detector(background, args.diff_threshold, args.min_area, adapt_rate=0.0)
    models = lt.load_models(args.models, work_size, names)

    samples = []   # (car, score, label)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    index = -1
    wanted = sorted(gt)
    while wanted and index < wanted[-1]:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index not in gt:
            continue
        small = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        _, blobs, scores, _ = lt.detect_frame(small, detector, models)
        for ci, name in enumerate(names):
            u, v = gt[index].get(name, (-1, -1))
            visible = u >= 0
            pu, pv = u / scale[0], v / scale[1]
            covered = False
            for bi, blob in enumerate(blobs):
                inside = (visible and blob.x - args.tolerance <= pu <= blob.x + blob.w + args.tolerance
                          and blob.y - args.tolerance <= pv <= blob.y + blob.h + args.tolerance)
                covered |= inside
                samples.append((name, scores[ci][bi], int(inside)))
            if visible and not covered:    # vom Vordergrund-Schritt verpasst
                samples.append((name, -np.inf, 1))
    cap.release()
    return names, samples


def roc(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, samples = collect_samples(args)
    fig, ax = plt.subplots(figsize=(6, 6))
    rows = []
    groups = [(n, [s for s in samples if s[0] == n]) for n in names] + [("Alle Autos", samples)]
    for name, group in groups:
        scores = [s[1] for s in group]
        labels = [s[2] for s in group]
        fpr, tpr, thresholds = roc_points(scores, labels)
        auc = float(_trapezoid(tpr, fpr))
        best = int(np.argmax(tpr - fpr))
        current = int(np.argmin(np.abs(np.where(np.isfinite(thresholds), thresholds, 9) - args.threshold)))
        style = dict(lw=2.5, color="black") if name == "Alle Autos" else dict(lw=1.5)
        ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})", **style)
        if name == "Alle Autos":
            ax.scatter([fpr[current]], [tpr[current]], s=60, zorder=5, color="red",
                       label=f"aktuelle Schwelle {args.threshold:.2f}")
        print(f"{name:15s} P={sum(labels):5d} N={len(labels) - sum(labels):5d} AUC={auc:.3f} | "
              f"Schwelle {args.threshold:.2f}: TPR={tpr[current]:.3f} FPR={fpr[current]:.3f} | "
              f"bestes Youden-J bei {thresholds[best]:.3f}: TPR={tpr[best]:.3f} FPR={fpr[best]:.3f}")
        rows += [[name, f"{th:.4f}", f"{f:.4f}", f"{t:.4f}"] for th, f, t in zip(thresholds, fpr, tpr)]
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC – Farbhistogramm-Klassifikator", xlim=(0, 1), ylim=(0, 1.02))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    with open(Path(args.out).with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["car_id", "threshold", "fpr", "tpr"])
        w.writerows(rows)
    print(f"ROC gespeichert: {args.out} (+ .csv)")


# ---------------------------------------------------------------------------
# 3) Mapping-Fehler
# ---------------------------------------------------------------------------

def angle_error(a, b, period=360.0):
    return abs((a - b + period / 2) % period - period / 2)


def mapping(args):
    with open(args.log, newline="", encoding="utf-8") as f:
        log = list(csv.DictReader(f))
    with open(args.truth, newline="", encoding="utf-8") as f:
        truth = list(csv.DictReader(f))
    out_rows, all_pos, all_th, all_axis = [], [], [], []
    print(f"{'Auto':14s} {'Soll x,y (mm)':>16s} {'Frames':>6s} {'Erk.':>5s} "
          f"{'Pos Ø':>7s} {'Pos max':>7s} {'θ Ø':>6s} {'θ max':>6s} {'Achse Ø':>7s}")
    for t in truth:
        t0, t1 = float(t["t_start_s"]) * 1e6, float(t["t_end_s"]) * 1e6
        x, y, th = float(t["x_mm"]), float(t["y_mm"]), float(t["theta_deg"])
        window = [r for r in log if r["car_id"] == t["car_id"] and t0 <= float(r["timestamp_us"]) <= t1]
        hits = [r for r in window if r["detected"] == "1"]
        if not hits:
            print(f"{t['car_id']:14s} {x:7.0f},{y:7.0f}  keine Erkennung im Zeitfenster")
            continue
        pos = np.array([math.hypot(float(r["x_mm"]) - x, float(r["y_mm"]) - y) for r in hits])
        thetas = [float(r["theta_deg"]) for r in hits if r["theta_deg"] != "nan"]
        th_err = np.array([angle_error(v, th) for v in thetas]) if thetas else np.array([np.nan])
        ax_err = np.array([angle_error(v, th, 180.0) for v in thetas]) if thetas else np.array([np.nan])
        all_pos += list(pos)
        all_th += list(th_err)
        all_axis += list(ax_err)
        rate = len(hits) / len(window)
        print(f"{t['car_id']:14s} {x:7.0f},{y:7.0f} {len(window):6d} {rate:5.0%} "
              f"{pos.mean():7.1f} {pos.max():7.1f} {np.nanmean(th_err):6.1f} {np.nanmax(th_err):6.1f} "
              f"{np.nanmean(ax_err):7.1f}")
        mean_x = np.mean([float(r["x_mm"]) for r in hits])
        mean_y = np.mean([float(r["y_mm"]) for r in hits])
        out_rows.append([t["car_id"], x, y, th, len(window), f"{rate:.3f}", f"{mean_x:.1f}", f"{mean_y:.1f}",
                         f"{pos.mean():.2f}", f"{pos.max():.2f}", f"{pos.std():.2f}",
                         f"{np.nanmean(th_err):.2f}", f"{np.nanmax(th_err):.2f}", f"{np.nanmean(ax_err):.2f}"])
    if not all_pos:
        raise SystemExit("Keine Messwerte – stimmen Namen und Zeitfenster?")
    print(f"\nGESAMT  Position: Ø {np.mean(all_pos):.1f} mm, max {np.max(all_pos):.1f} mm | "
          f"Orientierung: Ø {np.nanmean(all_th):.1f}°, max {np.nanmax(all_th):.1f}° "
          f"(nur Achse, ohne vorne/hinten: Ø {np.nanmean(all_axis):.1f}°)")
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["car_id", "true_x", "true_y", "true_theta", "frames", "detection_rate", "mean_x", "mean_y",
                    "pos_err_mean", "pos_err_max", "pos_err_std", "theta_err_mean", "theta_err_max",
                    "axis_err_mean"])
        w.writerows(out_rows)
    print(f"Tabelle gespeichert: {args.out}")


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("annotate", help="Ground Truth per Mausklick")
    a.add_argument("--video", required=True)
    a.add_argument("--cars", default="Red Racer,Green Hornet")
    a.add_argument("--step", type=int, default=10, help="jeden N-ten Frame annotieren")
    a.add_argument("--out", default="gt.csv")
    a.set_defaults(func=annotate)

    r = sub.add_parser("roc", help="ROC-Kurve")
    r.add_argument("--video", required=True)
    r.add_argument("--gt", required=True)
    r.add_argument("--cars", default="Red Racer,Green Hornet")
    r.add_argument("--models", default="car_models.json")
    r.add_argument("--background", default="background.png")
    r.add_argument("--bg-seconds", type=float, default=5.0)
    r.add_argument("--work-width", type=int, default=640)
    r.add_argument("--diff-threshold", type=int, default=30)
    r.add_argument("--min-area", type=int, default=30)
    r.add_argument("--threshold", type=float, default=0.45, help="Schwelle des Trackers (roter Punkt)")
    r.add_argument("--tolerance", type=float, default=4.0, help="Pixel-Toleranz um die Blob-Box")
    r.add_argument("--out", default="roc.png")
    r.set_defaults(func=roc)

    m = sub.add_parser("mapping", help="Mapping-Fehler stehender Autos")
    m.add_argument("--log", required=True, help="CSV aus live_tracker.py --log")
    m.add_argument("--truth", required=True, help="CSV mit gemessenen Soll-Positionen")
    m.add_argument("--out", default="mapping_errors.csv")
    m.set_defaults(func=mapping)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()