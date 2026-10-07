# robotics – Toy Race Car global vision

Two trackers share the same floor calibration (`calibration_floor.json`) and the same UDP format:

| File | Purpose |
|---|---|
| `live_tracker.py` | **Real-time tracker** (webcam or video): background subtraction + colour-histogram identity, homography to mm, Kalman velocity, heading, UDP output. ~4–5 ms per frame on a laptop CPU (budget at 60 FPS: 16.7 ms). |
| `sam_tracking.py` | Offline SAM2 tracker (high-quality masks, far from real time). Useful for comparison or to create reference data. |
| `evaluate.py` | Evaluation for the report: `annotate` (click ground truth), `roc` (ROC curve), `mapping` (position/orientation error of a non-moving car). |
| `server_udp.py`, `test_auto.py` | UDP publisher and `CarState` message format. |
| `test_receiver.py` | Prints received UDP packets. |
| `tools/make_synthetic_video.py` | Synthetic test video with exact ground truth (no real car needed). |

## Install

```bash
pip install -r requirements.txt          # live tracker + evaluation
# only for sam_tracking.py: pip install torch torchvision git+https://github.com/facebookresearch/sam2.git
```

## Run the live tracker

```bash
python test_receiver.py                                    # terminal 1: shows the UDP packets
python live_tracker.py --source 0 --cars "Red Racer,Green Hornet" --car-height 50   # terminal 2
```

First start (each step is saved and reused next time):

1. **Floor calibration** – click four floor points A-B-C-D of a known rectangle (`--tile-mm`, `--tiles-x`, `--tiles-y`, same dialog as `sam_tracking.py`). Redo with `--recalibrate`.
2. **Background** – clear the field and press **B** (video files: median of the first seconds). Redo with `--rebackground`.
3. **Colour models** – place the cars in view, press **SPACE**, draw a tight box around each car + Enter. Redo with `--reselect`.

Keys: **Q/Esc** quit, **B** new background (live), **SPACE** pause (video).

Useful options: `--port 5000`, `--udp-host <ip>`, `--threshold 0.45`, `--car-height <mm>`,
`--camera-pos x,y,h` (camera foot point and height in mm, measured with a tape – otherwise estimated from the homography with `--hfov`), `--log run.csv`, `--no-display`, `--display-every 2`.
On Windows the webcam is opened with DirectShow + MJPG, which most USB webcams need for 60 FPS.

Output per car and frame (UTF-8, UDP):
```
timestamp_us:"car id",x_mm,y_mm,theta_deg,vx_mm_s,vy_mm_s,omega_deg_s,u_px,v_px
816667:"Green Hornet",1739.401,1109.249,28.003,687.244,383.649,45.352,781,291
```
Not detected: `x,y = -1000.0,-1000.0`, other values `nan`, `u,v = -1`.
`theta` is measured from the +X towards the +Y axis of the calibrated floor rectangle.

## How it works

1. **Foreground**: |frame − background| > threshold on a 640 px wide working image, morphological cleanup.
2. **Candidates**: connected components of plausible size.
3. **Identity / classifier score**: Bhattacharyya similarity of the H/S histogram to each car's reference histogram. Greedy assignment, gated around the Kalman prediction. Touching cars (one merged blob) are split pixel-wise by colour back-projection.
4. **Position**: blob contour → homography → polygon centroid in mm, corrected for car height (the silhouette lies at ~half car height, so it is projected too far away from the camera; corrected with the intercept theorem using the camera position).
5. **Velocity**: constant-velocity Kalman filter in mm.
6. **Heading**: cars drive in the direction they face and cannot turn on the spot → while moving, theta = direction of motion; while standing, the last heading is held; a car that has never moved uses the silhouette axis (front/back unknown). Limitation: reversing is reported as driving forward in the opposite direction.
7. **Angular velocity**: smoothed derivative of theta.

## Evaluation

```bash
# ROC: annotate every 10th frame of a recorded video, then sweep the similarity threshold
python evaluate.py annotate --video test.mp4 --step 10 --out gt.csv
python evaluate.py roc --video test.mp4 --gt gt.csv --out roc.png

# Mapping error: record a video of non-moving cars at tape-measured positions
python live_tracker.py --source static.mp4 --no-udp --log static_log.csv
python evaluate.py mapping --log static_log.csv --truth positions.csv
```
`positions.csv`: `car_id,t_start_s,t_end_s,x_mm,y_mm,theta_deg` (video time in seconds, shown in the tracker window).

### Results on the synthetic video (`tools/make_synthetic_video.py`, 2 cars, 12 s @ 60 FPS)

| | Red Racer | Green Hornet |
|---|---|---|
| Detection rate | 98.5 % | 99.2 % |
| Position error moving (mean) | 12.7 mm | 10.9 mm |
| Heading error moving (mean) | 4.2° | 4.0° |
| Position error static (mean / max) | 12.0 / 12.0 mm | 12.1 / 12.2 mm |
| Heading error static | 0.8° | 3.0° |

Processing time 4.7 ms mean, 6 ms 95th percentile per frame (2-core cloud VM). Without the height correction the position error was ~70 mm; using the silhouette axis instead of the motion direction gave ~45° heading error.