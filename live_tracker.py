"""
Live Tracker for Toy Race Cars – fast real-time version of sam_tracking.py.

Pipeline per frame (on a smaller working image, default width: 640 px):

 1. Foreground      |Image - Background| > threshold (background = empty playing field)
 2. Candidates      connected foreground blobs with a reasonable size
 3. Identification  similarity of the H/S color histogram to the reference model of each car
                    (Bhattacharyya, 0..1) -> this is the "classifier score" for the ROC curve
 4. Position        contour -> homography -> polygon center in mm
 5. Direction       main axis of the polygon in mm; 180° ambiguity is solved using movement direction
 6. Speed           Kalman filter (constant velocity) in mm and mm/s
 7. Output          CarState -> UTF-8 -> UDP (format from the assignment)

The floor calibration (calibration_floor.json) is the same as in sam_tracking.py.

Examples:
    python live_tracker.py --source 0 --cars "Red Racer,Green Hornet"          # Webcam
    python live_tracker.py --source video.mp4 --cars "Red Racer,Green Hornet"  # Video
    python live_tracker.py --source 0 --udp-host 192.168.0.20 --port 5001      # different receiver
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from server_udp import UdpPublisher, valid_port
from test_auto import CarState

# Gleiche Kalibrierdatei und gleicher Klick-Dialog wie im SAM-Tracker.
from sam_tracking import load_or_select_calibration


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

HIST_BINS = [30, 16]              # Hue x Saturation
HIST_RANGES = [0, 180, 0, 256]
MIN_SATURATION = 50               # graue/weiße Bodenpixel nicht ins Farbmodell
MIN_VALUE = 40                    # sehr dunkle Pixel (Schatten) ignorieren
MAX_BLOBS = 10                    # nur die größten Kandidaten bewerten
AREA_RATIO_RANGE = (0.2, 5.0)     # erlaubte Fläche relativ zur Referenzfläche (Perspektive!)
GATE_MM = 300.0                   # Suchradius um die vorhergesagte Position
MAX_SPEED_MM_S = 6000.0           # 20 km/h ~ 5556 mm/s, plus Reserve
LOST_RESET_S = 0.5                # danach wird die Spur neu initialisiert
WINDOW = "Live Tracker"


# ---------------------------------------------------------------------------
# Farbmodell
# ---------------------------------------------------------------------------

def colour_histogram(hsv, mask):
    """Normiertes H/S-Histogramm der gesättigten Pixel unter mask (uint8, 0/255)."""
    saturated = cv2.inRange(hsv, (0, MIN_SATURATION, MIN_VALUE), (180, 255, 255))
    use = cv2.bitwise_and(mask, saturated)
    if cv2.countNonZero(use) < 15:
        return None
    hist = cv2.calcHist([hsv], [0, 1], use, HIST_BINS, HIST_RANGES)
    return (hist / max(float(hist.sum()), 1e-8)).astype(np.float32)


def similarity(a, b):
    """1 = identische Farbverteilung, 0 = keine Überlappung."""
    if a is None or b is None:
        return 0.0
    return float(1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


@dataclass
class CarModel:
    name: str
    hist: np.ndarray
    area: float  # Blob-Fläche bei der Auswahl, in Pixeln des Arbeitsbildes


def build_model(name, small, foreground, roi):
    """Farbmodell aus einem markierten Rechteck (x, y, w, h) im Arbeitsbild."""
    x, y, w, h = [int(v) for v in roi]
    if w <= 0 or h <= 0:
        raise ValueError(f"Leere Auswahl für {name}.")
    hsv = cv2.cvtColor(small[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
    mask = foreground[y:y + h, x:x + w] if foreground is not None else np.full((h, w), 255, np.uint8)
    hist = colour_histogram(hsv, mask)
    if hist is None:
        raise ValueError(f"Zu wenige farbige Pixel für {name}; enger/genauer markieren.")
    area = cv2.countNonZero(mask)
    return CarModel(name, hist, float(max(area, 20)))


def save_models(path, models, work_size):
    data = dict(schema_version=1, work_size=list(work_size), hist_bins=HIST_BINS,
                cars=[dict(name=m.name, area=m.area, hist=m.hist.flatten().tolist()) for m in models])
    Path(path).write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Farbmodelle gespeichert: {path}", flush=True)


def load_models(path, work_size, names):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("work_size") != list(work_size):
        raise ValueError("Farbmodelle passen nicht zur Arbeitsauflösung; mit --reselect neu auswählen.")
    stored = {c["name"]: c for c in data["cars"]}
    missing = [n for n in names if n not in stored]
    if missing:
        raise ValueError(f"Keine Farbmodelle für {missing}; mit --reselect neu auswählen.")
    return [CarModel(n, np.array(stored[n]["hist"], np.float32).reshape(HIST_BINS), stored[n]["area"])
            for n in names]


# ---------------------------------------------------------------------------
# Vordergrund und Kandidaten
# ---------------------------------------------------------------------------

@dataclass
class Blob:
    x: int
    y: int
    w: int
    h: int
    area: int
    cx: float          # Schwerpunkt im Arbeitsbild
    cy: float
    mask: np.ndarray   # uint8-Maske im Bounding-Box-Ausschnitt


class Detector:
    def __init__(self, background, diff_threshold=30, min_area=30, max_area=None, adapt_rate=0.002):
        self.background = background.astype(np.float32)
        self.background_u8 = background.copy()
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area or background.shape[0] * background.shape[1] // 4
        self.adapt_rate = adapt_rate
        self.k_open = np.ones((3, 3), np.uint8)
        self.k_close = np.ones((5, 5), np.uint8)
        self.k_protect = np.ones((15, 15), np.uint8)

    def foreground(self, small):
        diff = cv2.absdiff(small, self.background_u8)
        b, g, r = cv2.split(diff)
        strongest = cv2.max(cv2.max(b, g), r)
        _, fg = cv2.threshold(strongest, self.diff_threshold, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.k_open)
        return cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.k_close)

    def adapt(self, small, foreground):
        """Langsame Anpassung an Lichtänderungen – nur dort, wo kein Objekt ist."""
        if self.adapt_rate <= 0:
            return
        free = cv2.bitwise_not(cv2.dilate(foreground, self.k_protect))
        cv2.accumulateWeighted(small, self.background, self.adapt_rate, mask=free)
        self.background_u8 = cv2.convertScaleAbs(self.background)

    def blobs(self, foreground):
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        found = []
        for i in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[i])
            if not self.min_area <= area <= self.max_area:
                continue
            crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
            found.append(Blob(x, y, w, h, area, float(centroids[i][0]), float(centroids[i][1]), crop))
        found.sort(key=lambda blob: blob.area, reverse=True)
        return found[:MAX_BLOBS]


def blob_histogram(hsv, blob):
    crop = hsv[blob.y:blob.y + blob.h, blob.x:blob.x + blob.w]
    return colour_histogram(crop, blob.mask)


def blob_score(model, blob, hist):
    """Klassifikator-Score: Farbähnlichkeit, 0 wenn die Größe nicht passt."""
    ratio = blob.area / model.area
    if not AREA_RATIO_RANGE[0] <= ratio <= AREA_RATIO_RANGE[1]:
        return 0.0
    return similarity(model.hist, hist)


# ---------------------------------------------------------------------------
# Geometrie: Bild <-> Boden
# ---------------------------------------------------------------------------

def estimate_camera(homography, image_size, hfov_deg):
    """
    Schätzt Kameraposition über dem Boden aus der Homographie (Bild -> Boden).

    Annahme: Lochkamera mit Hauptpunkt in der Bildmitte und horizontalem
    Öffnungswinkel hfov_deg. Liefert (x, y) des Fußpunkts in mm und die Höhe in mm.
    """
    w, h = image_size
    f = (w / 2) / math.tan(math.radians(hfov_deg) / 2)
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ np.linalg.inv(np.asarray(homography, np.float64))
    M /= np.linalg.norm(M[:, 0])
    if M[2, 2] < 0:            # Bodenursprung muss vor der Kamera liegen
        M = -M
    r1, r2, t = M[:, 0], M[:, 1], M[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    centre = -R.T @ t
    return centre[:2], abs(float(centre[2]))


class FloorMapper:
    """Rechnet Pixel des Arbeitsbildes in Bodenkoordinaten (mm) um und zurück."""

    def __init__(self, homography, scale_xy, camera_ground=None, camera_height=None, car_height=0.0):
        self.H = np.asarray(homography, np.float64)
        self.H_inv = np.linalg.inv(self.H)
        self.scale = np.asarray(scale_xy, np.float64)   # Originalpixel je Arbeitspixel
        self.camera_ground = None if camera_ground is None else np.asarray(camera_ground, np.float64)
        self.camera_height = camera_height
        self.car_height = car_height

    def height_corrected(self, xy):
        """
        Die Homographie projiziert jeden Bildpunkt auf den Boden (z = 0). Der Umriss
        eines Autos liegt aber im Mittel auf halber Autohöhe und erscheint deshalb zu
        weit von der Kamera entfernt. Strahlensatz: X = C + (X' - C) * (Hc - z) / Hc.
        """
        if self.camera_ground is None or not self.car_height or not self.camera_height:
            return np.asarray(xy, np.float64)
        z = self.car_height / 2
        factor = (self.camera_height - z) / self.camera_height
        return self.camera_ground + (np.asarray(xy, np.float64) - self.camera_ground) * factor

    def to_mm(self, points_work):
        pts = (np.asarray(points_work, np.float64).reshape(-1, 2) * self.scale).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_work(self, points_mm):
        pts = np.asarray(points_mm, np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2) / self.scale


@dataclass
class Measurement:
    x: float
    y: float
    axis_deg: float      # Hauptachse in [-90, 90), ohne vorne/hinten
    elongation: float    # >1: länglich, ~1: rund -> Achse unzuverlässig
    u: float             # Schwerpunkt im Originalbild (Pixel)
    v: float
    score: float


def measure(blob, mapper, score):
    """Position und Achse des Autos in Bodenkoordinaten aus der Blob-Kontur."""
    contours, _ = cv2.findContours(blob.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    contour += (blob.x, blob.y)
    polygon = mapper.to_mm(contour).astype(np.float32)
    m = cv2.moments(polygon)
    u, v = blob.cx * mapper.scale[0], blob.cy * mapper.scale[1]
    if m["m00"] < 1.0:     # entartetes Polygon: nur Schwerpunkt projizieren
        x, y = mapper.height_corrected(mapper.to_mm([[blob.cx, blob.cy]])[0])
        return Measurement(float(x), float(y), 0.0, 1.0, u, v, score)
    x, y = mapper.height_corrected((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
    axis = 0.5 * math.degrees(math.atan2(2.0 * mu11, mu20 - mu02))
    common = math.sqrt(4.0 * mu11 ** 2 + (mu20 - mu02) ** 2)
    major = (mu20 + mu02 + common) / 2.0
    minor = max((mu20 + mu02 - common) / 2.0, 1e-9)
    return Measurement(float(x), float(y), axis, math.sqrt(major / minor), u, v, score)


# ---------------------------------------------------------------------------
# Zustandsschätzung: Kalman-Filter, Richtung, Winkelgeschwindigkeit
# ---------------------------------------------------------------------------

def wrap180(angle):
    return (angle + 180.0) % 360.0 - 180.0


class ConstantVelocityKalman:
    """Zustand [x, y, vx, vy] in mm bzw. mm/s, variable Zeitschritte."""

    def __init__(self, accel_std=5000.0, meas_std=10.0):
        self.q = accel_std ** 2
        self.r = meas_std ** 2
        self.x = None
        self.P = None

    def reset(self, z):
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([self.r, self.r, 1e6, 1e6])

    def predict(self, dt):
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.q * (G @ G.T)

    def update(self, z):
        S = self.P[:2, :2] + self.r * np.eye(2)
        K = self.P[:, :2] @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z) - self.x[:2])
        self.P = self.P - K @ self.P[:2, :]


class HeadingResolver:
    """
    Fahrtrichtung theta in [0, 360), gemessen von der +X- zur +Y-Achse des Feldes.

    Autos fahren in Blickrichtung und können sich im Stand nicht drehen. Deshalb:
      - in Fahrt (Tempo >= min_speed): theta = Bewegungsrichtung aus dem Kalman-Filter
      - im Stand: letzter Wert wird gehalten
      - noch nie gefahren: Hauptachse des Umrisses (vorne/hinten noch unbekannt)
    Grenze: Rückwärtsfahrt wird als Vorwärtsfahrt in Gegenrichtung gemeldet.
    """

    def __init__(self, min_speed=150.0, min_elongation=1.25, smoothing=0.3):
        self.min_speed = min_speed
        self.min_elongation = min_elongation
        self.smoothing = smoothing
        self.theta = None
        self.from_motion = False

    def reset(self):
        self.theta = None
        self.from_motion = False

    def update(self, axis, elongation, vx, vy):
        """Gibt (theta, jumped) zurück; jumped=True bei einem Sprung > 90°."""
        if math.hypot(vx, vy) >= self.min_speed:
            measured = math.degrees(math.atan2(vy, vx)) % 360.0
            first_motion = not self.from_motion
            self.from_motion = True
        elif self.from_motion:
            return self.theta, False
        elif elongation >= self.min_elongation:
            a = axis % 360.0
            b = (a + 180.0) % 360.0
            measured = a if self.theta is None or abs(wrap180(a - self.theta)) <= 90 else b
            first_motion = False
        else:
            return self.theta, False
        jumped = self.theta is None or first_motion or abs(wrap180(measured - self.theta)) > 90
        if jumped:
            self.theta = measured
        else:   # leichte Glättung, korrekt über den 0/360-Übergang
            self.theta = (self.theta + (1 - self.smoothing) * wrap180(measured - self.theta)) % 360.0
        return self.theta, jumped


class CarTrack:
    def __init__(self, model, args):
        self.model = model
        self.kf = ConstantVelocityKalman(args.accel_std, args.meas_std)
        self.heading = HeadingResolver(args.min_speed)
        self.last_seen = None
        self.last_time = None
        self.prev_theta = None
        self.prev_theta_time = None
        self.omega = 0.0

    @property
    def active(self):
        return self.kf.x is not None

    def reset(self):
        self.kf.x = None
        self.heading.reset()
        self.last_seen = self.prev_theta = None
        self.omega = 0.0

    def predict(self, t):
        if self.active and self.last_time is not None and t > self.last_time:
            self.kf.predict(t - self.last_time)
        self.last_time = t
        if self.active and t - self.last_seen > LOST_RESET_S:
            self.reset()

    def gate_ok(self, xy_mm, t):
        if not self.active:
            return True
        gate = GATE_MM + MAX_SPEED_MM_S * (t - self.last_seen)
        return float(np.hypot(*(np.asarray(xy_mm) - self.kf.x[:2]))) <= gate

    def correct(self, t, meas):
        if not self.active:
            self.kf.reset((meas.x, meas.y))
        else:
            self.kf.update((meas.x, meas.y))
        self.last_seen = t
        vx, vy = self.kf.x[2], self.kf.x[3]
        theta, flipped = self.heading.update(meas.axis_deg, meas.elongation, vx, vy)
        if theta is not None:
            if self.prev_theta is not None and not flipped and t > self.prev_theta_time:
                raw = wrap180(theta - self.prev_theta) / (t - self.prev_theta_time)
                self.omega = 0.7 * self.omega + 0.3 * raw
            self.prev_theta, self.prev_theta_time = theta, t
        return theta

    def state(self, timestamp_us, meas, theta):
        x, y, vx, vy = self.kf.x
        return CarState(timestamp_us=timestamp_us, car_id=self.model.name,
                        x=float(x), y=float(y),
                        theta=float(theta) if theta is not None else math.nan,
                        dx=float(vx), dy=float(vy), angular_velocity=float(self.omega),
                        u=int(round(meas.u)), v=int(round(meas.v)))


# ---------------------------------------------------------------------------
# Ein Frame verarbeiten (auch von evaluate.py genutzt)
# ---------------------------------------------------------------------------

def detect_frame(small, detector, models):
    """Gibt (Vordergrund, Blobs, Scores[car][blob], HSV-Bild) zurück."""
    foreground = detector.foreground(small)
    blobs = detector.blobs(foreground)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hists = [blob_histogram(hsv, blob) for blob in blobs]
    scores = [[blob_score(model, blob, hist) for blob, hist in zip(blobs, hists)] for model in models]
    return foreground, blobs, scores, hsv


def _sub_blob(blob, mask):
    """Größte Komponente einer Teilmaske als eigener Blob (Koordinaten im Arbeitsbild)."""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[i])
    if area < 15:
        return None
    crop = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
    return Blob(blob.x + x, blob.y + y, w, h, area,
                blob.x + float(centroids[i][0]), blob.y + float(centroids[i][1]), crop)


def split_blob(blob, hsv, model_a, model_b):
    """
    Trennt einen Blob, in dem zwei Autos verschmolzen sind, pixelweise nach Farbe:
    jedes Pixel geht an das Modell mit der höheren Rückprojektion (Back-Projection).
    """
    crop = hsv[blob.y:blob.y + blob.h, blob.x:blob.x + blob.w]
    probs = []
    for model in (model_a, model_b):
        hist = model.hist / max(float(model.hist.max()), 1e-8)
        probs.append(cv2.calcBackProject([crop], [0, 1], hist, HIST_RANGES, 255))
    inside = blob.mask > 0
    part_a = (inside & (probs[0] >= probs[1]) & (probs[0] > 0)).astype(np.uint8) * 255
    part_b = (inside & (probs[1] > probs[0])).astype(np.uint8) * 255
    a, b = _sub_blob(blob, part_a), _sub_blob(blob, part_b)
    return None if a is None or b is None else (a, b)


def assign(tracks, blobs, scores, mapper, t, threshold):
    """Greedy-Zuordnung: höchster Score zuerst, jedes Auto/Blob höchstens einmal."""
    if not blobs:
        return {}
    centres_mm = mapper.height_corrected(mapper.to_mm([[b.cx, b.cy] for b in blobs]))
    pairs = [(scores[ci][bi], ci, bi)
             for ci, track in enumerate(tracks) for bi in range(len(blobs))
             if scores[ci][bi] >= threshold and track.gate_ok(centres_mm[bi], t)]
    pairs.sort(reverse=True)
    used_cars, used_blobs, result = set(), set(), {}
    for score, ci, bi in pairs:
        if ci in used_cars or bi in used_blobs:
            continue
        used_cars.add(ci)
        used_blobs.add(bi)
        result[ci] = (bi, score)
    return result


def recover_merged(tracks, blobs, matches, hsv, mapper, t, threshold):
    """Fehlt ein verfolgtes Auto, wird geprüft, ob es mit einem anderen verschmolzen ist."""
    for ci, track in enumerate(tracks):
        if ci in matches or not track.active:
            continue
        for cj, (bi, score_j) in list(matches.items()):
            centre = mapper.height_corrected(mapper.to_mm([[blobs[bi].cx, blobs[bi].cy]])[0])
            if not track.gate_ok(centre, t):
                continue
            parts = split_blob(blobs[bi], hsv, tracks[cj].model, track.model)
            if parts is None:
                continue
            part_j, part_i = parts
            score_i = blob_score(track.model, part_i, blob_histogram(hsv, part_i))
            centre_i = mapper.height_corrected(mapper.to_mm([[part_i.cx, part_i.cy]])[0])
            if score_i < threshold or not track.gate_ok(centre_i, t):
                continue
            blobs.extend([part_j, part_i])
            matches[cj] = (len(blobs) - 2, score_j)
            matches[ci] = (len(blobs) - 1, score_i)
            break
    return matches


# ---------------------------------------------------------------------------
# Quelle, Hintergrund, Modelle (Start-Dialoge)
# ---------------------------------------------------------------------------

def open_source(source, width, height, fps):
    if source.isdigit():
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        # MJPG ist bei vielen USB-Webcams Voraussetzung für 60 FPS.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        live = True
    else:
        if not Path(source).is_file():
            raise SystemExit(f"Video nicht gefunden: {source}")
        cap = cv2.VideoCapture(source)
        live = False
    if not cap.isOpened():
        raise SystemExit(f"Quelle konnte nicht geöffnet werden: {source}")
    return cap, live


def work_size_for(frame, width):
    h, w = frame.shape[:2]
    width = min(width, w)
    height = max(2, round(h * width / w))
    return (width, height), (w / width, h / height)


def overlay_text(image, lines, origin=(10, 22)):
    for i, (text, colour) in enumerate(lines):
        pos = (origin[0], origin[1] + 20 * i)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)


def background_from_video(path, work_size, seconds):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or int(seconds * fps)
    count = min(count, max(1, int(seconds * fps)))
    samples = []
    for index in np.linspace(0, count - 1, 25).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA))
    cap.release()
    if not samples:
        raise SystemExit("Keine Frames für das Hintergrundbild.")
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def background_from_camera(cap, work_size):
    print("Spielfeld leer räumen, dann B drücken (Q = Abbruch).", flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("Kamera liefert keine Bilder.")
        small = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
        view = small.copy()
        overlay_text(view, [("Spielfeld leer raeumen, dann B druecken (Q = Abbruch)", (255, 255, 255))])
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt
        if key in (ord("b"), ord("B")):
            break
    samples = []
    for _ in range(30):
        ok, frame = cap.read()
        if ok:
            samples.append(cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA))
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


def select_models(cap, live, first_small, work_size, detector, names):
    small = first_small
    if live:
        print("Alle Autos ins Bild stellen, dann LEERTASTE drücken.", flush=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Kamera liefert keine Bilder.")
            small = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
            view = small.copy()
            overlay_text(view, [("Autos ins Bild stellen, dann LEERTASTE", (255, 255, 255))])
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise KeyboardInterrupt
            if key == ord(" "):
                break
    foreground = detector.foreground(small)
    models = []
    for name in names:
        print(f"'{name}' eng markieren und Enter drücken.", flush=True)
        roi = cv2.selectROI(f"{name} markieren (Enter)", small, False, False)
        cv2.destroyWindow(f"{name} markieren (Enter)")
        if roi[2] == 0 or roi[3] == 0:
            raise KeyboardInterrupt
        models.append(build_model(name, small, foreground, roi))
    return models


# ---------------------------------------------------------------------------
# Darstellung
# ---------------------------------------------------------------------------

PALETTE = [(0, 255, 0), (255, 0, 255), (0, 200, 255), (255, 200, 0), (0, 0, 255)]


def draw(view, tracks, results, mapper, fps, proc_ms, t):
    lines = [(f"t={t:7.2f} s | {fps:5.1f} FPS | Verarbeitung {proc_ms:5.1f} ms", (255, 255, 255))]
    for ci, track in enumerate(tracks):
        colour = PALETTE[ci % len(PALETTE)]
        hit = results.get(ci)
        if hit is None:
            lines.append((f"{track.model.name}: nicht gefunden", colour))
            continue
        blob, state = hit
        contours, _ = cv2.findContours(blob.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view, contours, -1, colour, 2, offset=(blob.x, blob.y))
        if not math.isnan(state.theta):
            rad = math.radians(state.theta)
            start, tip = mapper.to_work([[state.x, state.y],
                                         [state.x + 150 * math.cos(rad), state.y + 150 * math.sin(rad)]])
            cv2.arrowedLine(view, tuple(map(int, start)), tuple(map(int, tip)), colour, 2, tipLength=0.3)
        speed = math.hypot(state.dx, state.dy)
        lines.append((f"{track.model.name}: x={state.x:7.0f} y={state.y:7.0f} mm  "
                      f"th={state.theta:5.0f}  v={speed:6.0f} mm/s  w={state.angular_velocity:6.0f} deg/s",
                      colour))
    overlay_text(view, lines)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Echtzeit-Tracker für Toy Race Cars mit UDP-Ausgabe")
    p.add_argument("--source", default="0", help="Kameraindex (0, 1, ...) oder Videodatei")
    p.add_argument("--cars", default="Red Racer,Green Hornet", help="Namen der Autos, kommagetrennt")
    p.add_argument("--udp-host", default="127.0.0.1", help="IP des Empfängers")
    p.add_argument("--port", type=valid_port, default=5000, help="UDP-Zielport (Standard 5000)")
    p.add_argument("--no-udp", action="store_true", help="nichts senden")
    p.add_argument("--calibration", default="calibration_floor.json")
    p.add_argument("--tile-mm", type=float, default=600.0, help="nur für neue Kalibrierung")
    p.add_argument("--tiles-x", type=int, default=2)
    p.add_argument("--tiles-y", type=int, default=2)
    p.add_argument("--recalibrate", action="store_true")
    p.add_argument("--background", default="background.png", help="Hintergrundbild (Arbeitsauflösung)")
    p.add_argument("--rebackground", action="store_true", help="Hintergrund neu aufnehmen")
    p.add_argument("--bg-seconds", type=float, default=5.0, help="Video: Median über die ersten N s")
    p.add_argument("--models", default="car_models.json")
    p.add_argument("--reselect", action="store_true", help="Autos neu markieren")
    p.add_argument("--cam-width", type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--cam-fps", type=int, default=60)
    p.add_argument("--work-width", type=int, default=640, help="Breite des Arbeitsbildes")
    p.add_argument("--threshold", type=float, default=0.45, help="min. Farbähnlichkeit (0..1)")
    p.add_argument("--diff-threshold", type=int, default=30, help="Vordergrund-Schwelle (Grauwert)")
    p.add_argument("--min-area", type=int, default=30, help="min. Blob-Fläche (Arbeitspixel)")
    p.add_argument("--bg-adapt", type=float, default=0.002, help="Hintergrund-Lernrate (0 = aus)")
    p.add_argument("--accel-std", type=float, default=5000.0, help="Kalman: Beschleunigung mm/s²")
    p.add_argument("--meas-std", type=float, default=10.0, help="Kalman: Messrauschen mm")
    p.add_argument("--min-speed", type=float, default=150.0, help="ab hier gilt Fahrtrichtung (mm/s)")
    p.add_argument("--car-height", type=float, default=50.0, help="Autohöhe in mm (0 = keine Korrektur)")
    p.add_argument("--camera-pos", default=None,
                   help="Kamera-Fußpunkt und Höhe in mm 'x,y,h' (mit Maßband gemessen); sonst geschätzt")
    p.add_argument("--hfov", type=float, default=70.0, help="horiz. Öffnungswinkel der Kamera (Grad) für die Schätzung")
    p.add_argument("--log", default=None, help="CSV-Protokoll für die Auswertung")
    p.add_argument("--no-display", action="store_true", help="ohne Fenster (schneller)")
    p.add_argument("--display-every", type=int, default=1, help="nur jeden N-ten Frame anzeigen")
    p.add_argument("--max-frames", type=int, default=0, help="nach N Frames beenden (0 = nie)")
    p.add_argument(
        "--output-video",
        default=None,
        help="Annotiertes Ergebnisvideo speichern"
    )

    p.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Wiedergabegeschwindigkeit für Videodateien, z.B. 1.0 normal, 0.5 halb"
    )
    return p.parse_args()


def main():
    args = parse_args()
    names = [n.strip() for n in args.cars.split(",") if n.strip()]
    if not names:
        raise SystemExit("Mindestens ein Auto mit --cars angeben.")
    cap, live = open_source(args.source, args.cam_width, args.cam_height, args.cam_fps)
    publisher = None
    log_file = None
    video_writer = None
    try:
        ok, first = cap.read()
        if not ok:
            raise SystemExit("Kein erstes Bild.")
        h0, w0 = first.shape[:2]
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

        print(
            f"Quelle: {w0}x{h0} @ {video_fps:.1f} FPS "
            f"({'live' if live else 'Video'})",
            flush=True
        )

        work_size, scale = work_size_for(first, args.work_width)
        first_small = cv2.resize(
            first,
            work_size,
            interpolation=cv2.INTER_AREA
        )

        # FPS of saved annotated video
        output_fps = (
            float(args.cam_fps)
            if live
            else float(video_fps)
        )

        # Optional annotated video output
        if args.output_video:
            video_writer = cv2.VideoWriter(
                args.output_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                work_size
            )

            if not video_writer.isOpened():
                raise SystemExit(
                    f"Ausgabevideo konnte nicht geöffnet werden: "
                    f"{args.output_video}"
                )

            print(
                f"Ausgabevideo: {args.output_video} "
                f"@ {output_fps:.1f} FPS",
                flush=True
            )
        homography = load_or_select_calibration(first, Path(args.calibration).resolve(),
                                                args.tile_mm * args.tiles_x, args.tile_mm * args.tiles_y,
                                                args.recalibrate)
        if args.camera_pos:
            cx, cy, ch = (float(v) for v in args.camera_pos.split(","))
            camera_ground, camera_height = np.array([cx, cy]), ch
        else:
            camera_ground, camera_height = estimate_camera(homography, (w0, h0), args.hfov)
        print(f"Kamera: Fußpunkt ({camera_ground[0]:.0f}, {camera_ground[1]:.0f}) mm, "
              f"Höhe {camera_height:.0f} mm ({'gemessen' if args.camera_pos else 'geschätzt'})", flush=True)
        mapper = FloorMapper(homography, scale, camera_ground, camera_height, args.car_height)

        bg_path = Path(args.background)
        background = None
        if bg_path.is_file() and not args.rebackground:
            background = cv2.imread(str(bg_path))
            if background is None or background.shape[:2] != (work_size[1], work_size[0]):
                print("Gespeicherter Hintergrund passt nicht – wird neu erstellt.", flush=True)
                background = None
        if background is None:
            background = (background_from_camera(cap, work_size) if live
                          else background_from_video(args.source, work_size, args.bg_seconds))
            cv2.imwrite(str(bg_path), background)
            print(f"Hintergrund gespeichert: {bg_path}", flush=True)
        detector = Detector(background, args.diff_threshold, args.min_area, adapt_rate=args.bg_adapt)

        if Path(args.models).is_file() and not args.reselect:
            models = load_models(args.models, work_size, names)
            print(f"Farbmodelle geladen: {args.models}", flush=True)
        else:
            if args.no_display:
                raise SystemExit("Für die Auswahl der Autos wird ein Fenster gebraucht (ohne --no-display starten).")
            models = select_models(cap, live, first_small, work_size, detector, names)
            save_models(args.models, models, work_size)

        tracks = [CarTrack(model, args) for model in models]
        if not args.no_udp:
            publisher = UdpPublisher(args.udp_host, args.port)
            print(f"UDP an {publisher.destination[0]}:{publisher.destination[1]}", flush=True)
        if args.log:
            log_file = open(args.log, "w", newline="", encoding="utf-8")
            log = csv.writer(log_file)
            log.writerow(["frame", "timestamp_us", "car_id", "detected", "score", "x_mm", "y_mm",
                          "theta_deg", "vx_mm_s", "vy_mm_s", "omega_deg_s", "u", "v", "proc_ms"])

        if not live:   # Video von vorne, damit Zeitstempel = Videozeit
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        start_ns = time.perf_counter_ns()
        frame_index = 0
        fps_estimate = 0.0
        last_ns = start_ns
        proc_times = []
        paused = False
        print("Tracking läuft – Q/Esc beendet, B nimmt den Hintergrund neu auf (live).", flush=True)

        while True:
            if paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord(" "):
                    paused = False
                elif key in (ord("q"), 27):
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            now_ns = time.perf_counter_ns()
            timestamp_us = ((now_ns - start_ns) // 1000 if live
                            else round(frame_index / video_fps * 1_000_000))
            t = timestamp_us / 1e6
            dt_wall = (now_ns - last_ns) / 1e9
            last_ns = now_ns
            if dt_wall > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 / dt_wall if fps_estimate else 1 / dt_wall

            t0 = time.perf_counter()
            small = cv2.resize(frame, work_size, interpolation=cv2.INTER_AREA)
            foreground, blobs, scores, hsv = detect_frame(small, detector, models)
            if frame_index < 10:
             print(
                f"DEBUG frame={frame_index} "
                f"blobs={len(blobs)} "
                f"scores={scores}"
            )
            for track in tracks:
                track.predict(t)
            matches = assign(tracks, blobs, scores, mapper, t, args.threshold)
            if len(matches) < len(tracks):
                matches = recover_merged(tracks, blobs, matches, hsv, mapper, t, args.threshold)
            states, drawn = [], {}
            for ci, track in enumerate(tracks):
                if ci in matches:
                    bi, score = matches[ci]
                    meas = measure(blobs[bi], mapper, score)
                    theta = track.correct(t, meas)
                    state = track.state(timestamp_us, meas, theta)
                    drawn[ci] = (blobs[bi], state)
                else:
                    state, score = CarState.missing(timestamp_us, track.model.name), 0.0
                states.append((state, ci in matches, score))
            detector.adapt(small, foreground)
            proc_ms = (time.perf_counter() - t0) * 1000
            proc_times.append(proc_ms)

            for state, detected, score in states:
                if publisher is not None:
                    publisher.publish(state)
                if log_file is not None:
                    log.writerow([frame_index, state.timestamp_us, state.car_id, int(detected), f"{score:.4f}",
                                  f"{state.x:.2f}", f"{state.y:.2f}", f"{state.theta:.2f}",
                                  f"{state.dx:.1f}", f"{state.dy:.1f}", f"{state.angular_velocity:.1f}",
                                  state.u, state.v, f"{proc_ms:.2f}"])

            need_view = (
                video_writer is not None
                or (
                    not args.no_display
                    and frame_index % max(1, args.display_every) == 0
                )
            )

            if need_view:
                view = small.copy()

                draw(
                    view,
                    tracks,
                    drawn,
                    mapper,
                    fps_estimate,
                    proc_ms,
                    t
                )

                if video_writer is not None:
                    video_writer.write(view)

                if (
                    not args.no_display
                    and frame_index % max(1, args.display_every) == 0
                ):
                    cv2.imshow(WINDOW, view)

                    if live:
                        delay_ms = 1
                    else:
                        speed = max(args.playback_speed, 0.01)
                        target_ms = 1000.0 / video_fps / speed
                        elapsed_ms = (
                            time.perf_counter() - t0
                        ) * 1000.0
                        delay_ms = max(
                            1,
                            int(target_ms - elapsed_ms)
                        )

                    key = cv2.waitKey(delay_ms) & 0xFF

                    if key in (ord("q"), 27):
                        break

                    if key == ord(" ") and not live:
                        paused = True

                    if key in (ord("b"), ord("B")) and live:
                        background = background_from_camera(
                            cap,
                            work_size
                        )

                        cv2.imwrite(
                            str(bg_path),
                            background
                        )

                        detector = Detector(
                            background,
                            args.diff_threshold,
                            args.min_area,
                            adapt_rate=args.bg_adapt
                        )
                frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break

        if proc_times:
            arr = np.array(proc_times)
            print(f"{frame_index} Frames | Verarbeitung pro Frame: Mittel {arr.mean():.2f} ms, "
                  f"95%-Perzentil {np.percentile(arr, 95):.2f} ms, Max {arr.max():.2f} ms "
                  f"(Budget bei 60 FPS: 16.67 ms)", flush=True)
    except KeyboardInterrupt:
        print("Abgebrochen.", flush=True)
    finally:
        cap.release()
        if publisher is not None:
            publisher.close()
        if log_file is not None:
            log_file.close()
            print(f"Protokoll gespeichert: {args.log}", flush=True)
        if video_writer is not None:
            video_writer.release()
            print(
                f"Video gespeichert: {args.output_video}",
                flush=True
            )
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())