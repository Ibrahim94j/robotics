print("SAM tracking started", flush=True)

import os
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import argparse
import csv
import gc
import json
import socket
import time
import inspect
from contextlib import contextmanager
from pathlib import Path
import tempfile

print("Loading OpenCV and NumPy ...", flush=True)
import cv2
import numpy as np

COLORS = {1: (0, 255, 0), 2: (255, 0, 255)}


# Keep the aspect ratio and return the pixel scale for both axes.
def resize(frame, width):
    h, w = frame.shape[:2]
    scale = min(1.0, width / w)
    size = (max(2, round(w * scale) // 2 * 2),
            max(2, round(h * scale) // 2 * 2))
    return cv2.resize(frame, size), (size[0] / w, size[1] / h)


# Offline median background. This uses frames from the selected video interval.
def background_image(path, width, seconds):
    cap = cv2.VideoCapture(str(path))
    samples = []
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError('Invalid video frame rate.')
        count = min(count, max(1, int(seconds * fps)))
        for index in np.linspace(0, max(0, count - 1), 25).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                samples.append(resize(frame, width)[0])
    finally:
        cap.release()
    if not samples:
        raise RuntimeError('Could not read background frames.')
    return np.median(np.stack(samples), axis=0).astype(np.uint8)


# Compare saturated vehicle colours rather than the grey floor.
def profile(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    use = mask & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 30)
    if use.sum() < 8:
        return None
    hist = cv2.calcHist([hsv], [0, 1], use.astype(np.uint8) * 255,
                       [18, 8], [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (3, 3), 0)
    return (hist / max(float(hist.sum()), 1e-8)).astype(np.float32)


# A higher score means the colour histograms are more similar.
def similarity(a, b):
    if a is None or b is None:
        return 0.0
    return 1.0 - cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)


# Find foreground regions with a plausible size for a car.
def candidates(frame, background, area):
    diff = cv2.absdiff(frame, background).max(axis=2)
    foreground = (diff > 30).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                 np.ones((3, 3), np.uint8))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                 np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
    result = []
    for index in range(1, n):
        x, y, w, h, pixels = stats[index]
        if 0.15 * area <= pixels <= 5 * area:
            result.append(([max(0, x - 5), max(0, y - 5),
                            min(frame.shape[1], x + w + 5),
                            min(frame.shape[0], y + h + 5)],
                           labels == index))
    return result


# Convert model outputs into one boolean mask per car.
def decode(ids, logits, shape):
    result = {i: np.zeros(shape, bool) for i in (1, 2)}
    for index, car_id in enumerate(ids):
        if car_id in result:
            result[car_id] = (logits[index, 0] > 0).cpu().numpy()
    return result


# These bounds allow size changes and partially visible cars.
def valid(mask, area):
    return 0.1 * area <= int(mask.sum()) <= 5 * area


# Reject cases where both identities cover the same object.
def overlap(first, second):
    return np.count_nonzero(first & second) / max(1, min(first.sum(), second.sum()))



# Map original image pixels to the floor plane, including outside the reference rectangle.
def pixel_to_mm(matrix, u, v):
    homogeneous = matrix @ np.array([u, v, 1.0], np.float64)
    if not np.isfinite(homogeneous).all() or abs(homogeneous[2]) < 1e-9:
        return None
    return homogeneous[:2] / homogeneous[2]


# The clicked rectangle defines the origin and the positive X/Y directions.
def calibration_matrix(points, width_mm, height_mm):
    points = np.asarray(points, np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError('Four valid image points are required.')
    if not np.isfinite([width_mm, height_mm]).all() or min(width_mm, height_mm) <= 0:
        raise ValueError('Rectangle dimensions must be positive.')
    contour = points.reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour) or cv2.contourArea(contour) < 100:
        raise ValueError('Select a large enough convex quadrilateral in boundary order.')
    target = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]], np.float32)
    matrix = cv2.getPerspectiveTransform(points, target)
    if not np.isfinite(matrix).all() or np.linalg.matrix_rank(matrix) != 3:
        raise ValueError('Invalid calibration.')
    return matrix


# Save calibration in original pixel coordinates so processing width can change.
def load_or_select_calibration(original, path, width_mm, height_mm, recalibrate):
    h, w = original.shape[:2]
    if path.is_file() and not recalibrate:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or data.get('image_size') != [w, h]:
            raise ValueError('Calibration image size does not match; use --recalibrate.')
        if data.get('coordinate_space') != 'original_image_pixels':
            raise ValueError('Unknown calibration coordinate space.')
        matrix = calibration_matrix(data['image_points'], data['width_mm'], data['height_mm'])
        print(f"Calibration loaded: {path} ({data['width_mm']:g} x {data['height_mm']:g} mm)", flush=True)
        print('Reuse only with the same camera position and image framing.', flush=True)
        return matrix
    display, scales = resize(original, 1200)
    points = []
    labels = ['A: origin', 'B: along +X', 'C: opposite corner (+X,+Y)', 'D: along +Y']
    window = 'Floor calibration'
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    print(f'Floor rectangle: {width_mm:g} x {height_mm:g} mm', flush=True)
    print('Click four floor intersections A-B-C-D around the rectangle. Enter confirms; R resets; Esc cancels.', flush=True)
    try:
        while True:
            canvas = display.copy()
            instruction = labels[len(points)] if len(points) < 4 else 'Enter: confirm | R: reset'
            cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 56), (0, 0, 0), -1)
            cv2.putText(canvas, instruction, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(canvas, 'Left: add point | Right: undo | Esc: cancel', (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            for i, point in enumerate(points):
                cv2.circle(canvas, point, 5, (0, 255, 255), -1)
                cv2.putText(canvas, 'ABCD'[i], (point[0]+8, point[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if len(points) > 1:
                cv2.polylines(canvas, [np.array(points, np.int32)], len(points)==4, (0, 255, 255), 2)
            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 255
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt
            if key in (ord('r'), ord('R')):
                points.clear()
            if key in (10, 13) and len(points) == 4:
                original_points = np.asarray(points, np.float32) / np.asarray(scales, np.float32)
                try:
                    matrix = calibration_matrix(original_points, width_mm, height_mm)
                except ValueError as error:
                    print(error, flush=True)
                    points.clear()
                    continue
                break
    finally:
        cv2.destroyAllWindows()
    data = dict(schema_version=1, image_size=[w, h], coordinate_space='original_image_pixels',
                image_points=original_points.tolist(), width_mm=width_mm, height_mm=height_mm,
                homography=matrix.tolist(), note='Floor plane; provisional dimensions. Millimetre accuracy is not guaranteed.')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
    print(f'Calibration saved: {path}', flush=True)
    return matrix


# Hide the per-frame SAM progress bar while leaving other messages visible.
@contextmanager
def quiet_propagation(predictor):
    namespace = inspect.unwrap(predictor.propagate_in_video).__globals__
    original_tqdm = namespace.get('tqdm')
    if original_tqdm is None:
        yield
        return

    def quiet_tqdm(*args, **kwargs):
        kwargs['disable'] = True
        return original_tqdm(*args, **kwargs)

    namespace['tqdm'] = quiet_tqdm
    try:
        yield
    finally:
        namespace['tqdm'] = original_tqdm


def format_duration(seconds):
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
    return f'{minutes:02d}:{seconds:02d}'


# Count output frames, including the first frame and any skipped source frames.
class ProcessingProgress:
    def __init__(self, total):
        self.total = total
        self.started = time.monotonic()
        self.last_report = self.started
        self.last_count = -1
        self.update(0, force=True)

    def update(self, completed, force=False):
        now = time.monotonic()
        if not force and now - self.last_report < 5:
            return
        elapsed = now - self.started
        remaining = '--:--'
        if completed and self.total is not None:
            remaining = format_duration(elapsed / completed * max(0, self.total - completed))
        if self.total is None:
            count = f'{completed}/? frames'
        else:
            count = f'{completed}/{self.total} frames ({min(100, 100 * completed / self.total):.0f}%)'
        print(f'Progress: {count} | Elapsed: {format_duration(elapsed)} | Estimated remaining: {remaining}', flush=True)
        self.last_report = now
        self.last_count = completed

    def finish(self, completed):
        # The decoder reaching EOF gives the actual count if metadata was inaccurate.
        self.total = completed
        self.update(completed, force=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='output2_fixed.mp4')
    parser.add_argument('--checkpoint', default='weights/sam2.1_hiera_tiny.pt')
    parser.add_argument('--output', default='sam_calibrated.mp4')
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--seconds', type=float, default=10.0,
                        help='Video duration to process; default: 10 seconds')
    parser.add_argument('--step', type=int, default=4)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--chunk-frames', type=int, default=50)
    parser.add_argument('--reid-threshold', type=float, default=0.55)
    parser.add_argument('--udp-host', default=None, help='UDP receiver address; omit to disable UDP')
    parser.add_argument('--port', type=int, default=5000, help='UDP destination port')
    parser.add_argument('--calibration', default='calibration_floor.json')
    parser.add_argument('--tile-mm', type=float, default=600.0)
    parser.add_argument('--tiles-x', type=int, default=2)
    parser.add_argument('--tiles-y', type=int, default=2)
    parser.add_argument('--recalibrate', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be between 1 and 65535.')
    if not np.isfinite(args.tile_mm) or args.tile_mm <= 0 or min(args.tiles_x, args.tiles_y) < 1:
        parser.error('tile-mm must be > 0; tiles-x/tiles-y must be >= 1.')
    if args.step < 1 or args.width < 2 or args.chunk_frames < 1:
        parser.error('step/chunk-frames must be >= 1; width must be >= 2.')
    if not np.isfinite(args.reid_threshold) or not 0 <= args.reid_threshold <= 1:
        parser.error('reid-threshold must be between 0 and 1.')

    print('Loading PyTorch and SAM ...', flush=True)
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('seconds must be finite and greater than 0.')
    video = Path(args.video).resolve()
    weights = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    csv_path = output.with_suffix('.csv')
    calibration_path = Path(args.calibration).resolve()
    candidate_path = output.with_name(output.stem + '_candidates.csv')
    if len({video, weights, output, csv_path, calibration_path, candidate_path}) != 6:
        parser.error('Input, checkpoint, calibration and output paths must be different.')
    if not video.is_file() or not weights.is_file():
        parser.error('Video or checkpoint file is missing.')
    if video in (output, csv_path):
        parser.error('Input must not be overwritten.')
    if args.device == 'mps' and not torch.backends.mps.is_available():
        parser.error('MPS is unavailable; use --device cpu.')
    output.parent.mkdir(parents=True, exist_ok=True)
    print('Building background model for the fixed camera ...', flush=True)
    background = background_image(video, args.width, args.seconds)
    cap = cv2.VideoCapture(str(video))
    writer = None
    udp_socket = None
    previous_positions = {1: None, 2: None}
    try:
        if args.udp_host:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_target = (socket.gethostbyname(args.udp_host), args.port)
            print(f'UDP enabled: {udp_target[0]}:{args.port}', flush=True)
            print('Fields: timestamp_us, car_id, x_mm, y_mm, theta_deg, vx_mm_s, vy_mm_s, omega_deg_s, u_px, v_px', flush=True)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise RuntimeError('Invalid video frame rate.')
        ok, original = cap.read()
        if not ok:
            raise RuntimeError('Could not read the first frame.')
        first, scales = resize(original, args.width)
        matrix_mm = load_or_select_calibration(original, calibration_path,
            args.tile_mm * args.tiles_x, args.tile_mm * args.tiles_y, args.recalibrate)
        boxes = {}
        for car_id in (1, 2):
            print(f'Select car {car_id} closely and press Enter.')
            x, y, w, h = cv2.selectROI(f'Auto {car_id}', first, False, False)
            cv2.destroyAllWindows()
            if w == 0 or h == 0:
                return 0
            boxes[car_id] = np.array([x, y, x + w, y + h], np.float32)

        predictor = build_sam2_video_predictor(
            'configs/sam2.1/sam2.1_hiera_t.yaml', str(weights),
            device=args.device, apply_postprocessing=False)
        # Share the model, but use separate image features for reacquisition.
        image_predictor = SAM2ImagePredictor(predictor, max_hole_area=0, max_sprinkle_area=0)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps / args.step, (first.shape[1], first.shape[0]))
        if not writer.isOpened():
            raise RuntimeError('Could not open VideoWriter.')
        references, areas = {}, {}
        trails = {1: [], 2: []}
        pending = {1: None, 2: None}
        anchor = (0, first)
        anchor_masks = None
        original_index = 0
        processed = 0
        done = False
        print(f'Processing {args.seconds:g} seconds of video', flush=True)
        source_limit = max(1, round(args.seconds * fps))
        source_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = None
        if np.isfinite(source_count) and source_count >= 1:
            selected_count = min(source_limit, int(source_count))
            total_frames = (selected_count - 1) // args.step + 1
        progress = ProcessingProgress(total_frames)
        candidate_path = output.with_name(output.stem + '_candidates.csv')

        with csv_path.open('w', newline='', encoding='utf-8') as file, \
             candidate_path.open('w', newline='', encoding='utf-8') as candidate_file:
            candidate_writer = csv.writer(candidate_file)
            candidate_writer.writerow(['source_frame_number', 'video_time_s', 'car_id', 'candidate_rank', 'mask_index',
                'box_x1', 'box_y1', 'box_x2', 'box_y2', 'foreground_similarity',
                'mask_area', 'area_ratio', 'sam_quality', 'own_similarity',
                'other_similarity', 'identity_margin', 'accepted_by_filters', 'reason'])
            csv_writer = csv.writer(file)
            csv_writer.writerow(['source_frame_number', 'output_frame_number', 'video_time_s', 'car_id', 'detected', 'u', 'v',
                                 'mode', 'candidate_count', 'best_similarity', 'reason',
                                 'temporal_mask_area', 'fresh_mask_area', 'area_ratio',
                                 'fresh_own_similarity', 'fresh_other_similarity', 'identity_margin',
                                 'x_mm', 'y_mm', 'mapping_valid', 'vx_mm_s', 'vy_mm_s'])
            while not done:
                chunk = [anchor]
                while len(chunk) < args.chunk_frames + 1:
                    ok, image = cap.read()
                    if not ok:
                        done = True
                        break
                    original_index += 1
                    if original_index >= max(1, round(args.seconds * fps)):
                        done = True
                        break
                    if original_index % args.step == 0:
                        chunk.append((original_index, resize(image, args.width)[0]))
                if processed and len(chunk) == 1:
                    break
                state = None
                with tempfile.TemporaryDirectory(prefix='sam_') as folder:
                    for index, (_, image) in enumerate(chunk):
                        if not cv2.imwrite(str(Path(folder) / f'{index:05d}.jpg'), image):
                            raise RuntimeError('Could not save frame.')
                    try:
                        with torch.inference_mode():
                            state = predictor.init_state(folder, offload_video_to_cpu=True,
                                                         offload_state_to_cpu=True)
                            for car_id in (1, 2):
                                if anchor_masks is None:
                                    _, ids, logits = predictor.add_new_points_or_box(
                                        state, frame_idx=0, obj_id=car_id, box=boxes[car_id])
                                else:
                                    predictor.add_new_mask(state, frame_idx=0, obj_id=car_id,
                                                           mask=anchor_masks[car_id])
                            start = 0 if not processed else 1
                            for local in range(start, len(chunk)):
                                video_index, image = chunk[local]
                                # Process this frame while keeping the video state.
                                generator = predictor.propagate_in_video(
                                    state, start_frame_idx=local, max_frame_num_to_track=0)
                                with quiet_propagation(predictor):
                                    try:
                                        _, ids, logits = next(generator)
                                    finally:
                                        generator.close()
                                masks = decode(ids, logits, image.shape[:2])
                                if not references:
                                    for car_id in (1, 2):
                                        areas[car_id] = int(masks[car_id].sum())
                                        references[car_id] = profile(image, masks[car_id])
                                        if areas[car_id] < 10 or references[car_id] is None:
                                            raise RuntimeError('Invalid initial mask or colour profile; select a tighter box.')
                                accepted = {}
                                rows = {}
                                debug = {}
                                search_boxes = []
                                image_features_ready = False
                                for car_id in (1, 2):
                                    mask = masks[car_id]
                                    count, best = 0, 0.0
                                    mode, reason = 'SAM', 'ok'
                                    temporal_area = int(mask.sum())
                                    detail = [temporal_area, '', '', '', '', '']
                                    # Check identity again after a lost detection.
                                    tracked = valid(mask, areas[car_id]) and pending[car_id] is None
                                    if not tracked:
                                        mode, reason = 'Searching', 'no matching candidate'
                                        choices = candidates(image, background, areas[car_id])
                                        count = len(choices)
                                        ranked = []
                                        other = 3 - car_id
                                        for box, foreground_mask in choices:
                                            hist = profile(image, foreground_mask)
                                            own = similarity(references[car_id], hist)
                                            rival = similarity(references[other], hist)
                                            best = max(best, own)
                                            if own >= args.reid_threshold and own - rival >= 0.04:
                                                ranked.append((own, box))
                                        ranked.sort(key=lambda item: item[0], reverse=True)
                                        mask = np.zeros(image.shape[:2], bool)
                                        previous = pending[car_id]
                                        pending[car_id] = np.array([-10000., -10000.])
                                        if ranked:
                                            if not image_features_ready:
                                                image_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                                                image_features_ready = True
                                            successful = []
                                            failures = []
                                            # Test several candidate boxes and mask proposals.
                                            for rank, (score, box) in enumerate(ranked[:3], 1):
                                                fresh_masks, quality, _ = image_predictor.predict(
                                                    box=np.array(box, np.float32), multimask_output=True)
                                                box_ok = False
                                                for mask_index, fresh in enumerate(fresh_masks):
                                                    trial = fresh.astype(bool)
                                                    area = int(trial.sum())
                                                    ratio = area / max(1, areas[car_id])
                                                    hist = profile(image, trial)
                                                    own = similarity(references[car_id], hist)
                                                    rival = similarity(references[other], hist)
                                                    why = []
                                                    if not valid(trial, areas[car_id]):
                                                        why.append('MaskSize')
                                                    if own < args.reid_threshold:
                                                        why.append('ColourProfile')
                                                    if own - rival < 0.04:
                                                        why.append('IdentityMargin')
                                                    eligible = not why
                                                    candidate_writer.writerow([
                                                        video_index + 1, f'{video_index / fps:.6f}', car_id, rank, mask_index,
                                                        *box, f'{score:.3f}', area, f'{ratio:.3f}',
                                                        f'{float(quality[mask_index]):.3f}', f'{own:.3f}',
                                                        f'{rival:.3f}', f'{own - rival:.3f}', int(eligible),
                                                        '|'.join(why) if why else 'ok'])
                                                    measurement = [temporal_area, area, f'{ratio:.3f}',
                                                        f'{own:.3f}', f'{rival:.3f}', f'{own - rival:.3f}']
                                                    if eligible:
                                                        box_ok = True
                                                        successful.append((own, trial, measurement))
                                                    else:
                                                        failures.append((own, '|'.join(why), measurement))
                                                search_boxes.append((car_id, box, box_ok))
                                            if successful:
                                                successful.sort(key=lambda item: item[0], reverse=True)
                                                _, trial, detail = successful[0]
                                                ys, xs = np.nonzero(trial)
                                                center = np.array([xs.mean(), ys.mean()])
                                                if previous is not None and np.linalg.norm(center - previous) < 100:
                                                    mask = trial
                                                    pending[car_id] = None
                                                    # Replace the lost video prompt with the new mask.
                                                    predictor.add_new_mask(state, frame_idx=local,
                                                                           obj_id=car_id, mask=trial)
                                                    mode, reason = 'Reacquired', 'confirmed in two frames'
                                                else:
                                                    pending[car_id] = center
                                                    reason = 'waiting for second confirmation'
                                            elif failures:
                                                _, reason, detail = max(failures, key=lambda item: item[0])
                                    accepted[car_id] = mask
                                    rows[car_id] = [mode, count, best, reason]
                                    debug[car_id] = detail
                                if overlap(accepted[1], accepted[2]) > 0.3:
                                    for car_id in (1, 2):
                                        accepted[car_id] = np.zeros(image.shape[:2], bool)
                                        pending[car_id] = np.array([-10000., -10000.])
                                        rows[car_id][3] = 'IdentityConflict'
                                # Do not propagate rejected masks as valid detections.
                                for car_id in (1, 2):
                                    if not accepted[car_id].any():
                                        predictor.add_new_mask(state, frame_idx=local, obj_id=car_id,
                                                               mask=accepted[car_id])
                                result = image.copy()
                                for car_id, box, eligible in search_boxes:
                                    x1, y1, x2, y2 = map(int, box)
                                    color = COLORS[car_id] if eligible else (0, 165, 255)
                                    cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)
                                    cv2.putText(result, f'ID {car_id} Candidate', (x1, max(12, y1 - 4)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                                for car_id, mask in accepted.items():
                                    mode, count, best, reason = rows[car_id]
                                    ys, xs = np.nonzero(mask)
                                    detected = len(xs) > 0
                                    u = v = -1.0
                                    color = COLORS[car_id]
                                    if detected:
                                        u, v = float(xs.mean()), float(ys.mean())
                                        center = (round(u), round(v))
                                        trails[car_id].append(center)
                                        trails[car_id] = trails[car_id][-150:]
                                        result[mask] = (0.6 * result[mask] + 0.4 * np.array(color)).astype(np.uint8)
                                        cv2.circle(result, center, 4, color, -1)
                                    else:
                                        trails[car_id].clear()
                                    if len(trails[car_id]) > 1:
                                        cv2.polylines(result, [np.array(trails[car_id], np.int32)], False, color, 2)
                                    cv2.putText(result, f'Car {car_id}: {mode} / {reason}',
                                                (10, 28 * car_id), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                    # The mask centre is projected onto the floor; car height is not corrected.
                                    position_mm = pixel_to_mm(matrix_mm, u / scales[0], v / scales[1]) if detected else None
                                    if position_mm is not None:
                                        cv2.putText(result, f'Car {car_id}: X {position_mm[0]:.1f} mm / Y {position_mm[1]:.1f} mm',
                                                    (10, 84 + 24 * car_id), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                                    else:
                                        # -1000.0 signals a missing position; it is not a measured coordinate.
                                        cv2.putText(result, f'Car {car_id}: X -1000.0 mm / Y -1000.0 mm (invalid)',
                                                    (10, 84 + 24 * car_id), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
                                    # Use video time, not processing time, for motion estimates.
                                    timestamp_s = video_index / fps
                                    vx = vy = -1000.0
                                    previous_position = previous_positions[car_id]
                                    if position_mm is not None:
                                        if previous_position is not None:
                                            dt = timestamp_s - previous_position[0]
                                            if dt > 0:
                                                vx, vy = (position_mm - previous_position[1]) / dt
                                        previous_positions[car_id] = (timestamp_s, position_mm.copy())
                                    else:
                                        previous_positions[car_id] = None
                                    if udp_socket is not None:
                                        # Orientation is not implemented yet. Unknown fields use -1000.0.
                                        x_mm, y_mm = position_mm if position_mm is not None else (-1000.0, -1000.0)
                                        u_px = u / scales[0] if detected else -1000.0
                                        v_px = v / scales[1] if detected else -1000.0
                                        packet = (f'{round(timestamp_s * 1_000_000)},"{car_id}",'
                                                  f'{x_mm:.3f},{y_mm:.3f},-1000.0,'
                                                  f'{vx:.3f},{vy:.3f},-1000.0,'
                                                  f'{u_px:.3f},{v_px:.3f}\n')
                                        udp_socket.sendto(packet.encode('utf-8'), udp_target)
                                    csv_writer.writerow([video_index + 1, processed + 1, f'{video_index / fps:.6f}', car_id, int(detected),
                                                         f'{u / scales[0]:.2f}' if detected else -1,
                                                         f'{v / scales[1]:.2f}' if detected else -1,
                                                         mode, count, f'{best:.3f}', reason, *debug[car_id],
                                                         f'{position_mm[0]:.3f}' if position_mm is not None else '-1000.0',
                                                         f'{position_mm[1]:.3f}' if position_mm is not None else '-1000.0',
                                                         int(position_mm is not None), f'{vx:.3f}', f'{vy:.3f}'])
                                cv2.putText(result, f'Source frame {video_index + 1} | Output frame {processed + 1} | Time {video_index / fps:.3f} s',
                                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
                                cv2.putText(result, f'Source frame {video_index + 1} | Output frame {processed + 1} | Time {video_index / fps:.3f} s',
                                            (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                                writer.write(result)
                                processed += 1
                                anchor_masks = accepted
                                progress.update(processed)
                            anchor = chunk[-1]
                            file.flush()
                            candidate_file.flush()
                    finally:
                        if state is not None:
                            predictor.reset_state(state)
                        del state
                        gc.collect()
                        if args.device == 'mps':
                            torch.mps.empty_cache()
        progress.finish(processed)
        writer.release()
        writer = None
        for path in (output, csv_path, candidate_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f'Output is missing or empty: {path}')
            print(f'Saved: {path} ({path.stat().st_size} Bytes)')
    finally:
        cap.release()
        if udp_socket is not None:
            udp_socket.close()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('Cancelled; output written so far is kept.')
        raise SystemExit(130)
    except Exception:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
