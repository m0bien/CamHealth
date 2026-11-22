# CamHealth

Camera-based heart-rate monitoring in Julia: remote photoplethysmography (rPPG) from the
tiny colour changes blood flow causes in facial skin, plus optional MIT Eulerian Video
Magnification (EVM) to make the pulse visible. Works live from a webcam (Logitech C930e)
or offline from a video file. See [PLAN.md](PLAN.md) for the full roadmap.

**Status: M1 (offline baseline)** — lossless recording with locked camera settings,
YuNet face detection + landmark ROI, green-channel rPPG, Welch heart-rate estimate, CLI.

## Layout
```
src/CamHealth.jl                 main module
src/io/sources.jl                CameraSource / FileSource -> (frame, t); FrameMonitor (drops,
                                 brightness drift); lock_camera (Windows DirectShow)
src/filters/TemporalFilters.jl   standalone: BandSpec, PULSE, BREATHING, Butterworth band-pass
                                 (batch filtfilt + streaming step!), DiffLowpass (EVM), ideal FFT band-pass
src/face/face.jl                 YuNet (ONNX Runtime) detection, forehead + cheek ROI ∩ skin mask,
                                 FaceTracker (detect every 10 frames, EMA smoothing), FixedROI fallback
src/signal/extract.jl            ROI mean RGB per frame, resampling to a uniform grid, lost-face marking
src/signal/rppg.jl               green method (CHROM / POS in M2)
src/hr/estimate.jl               detrend, band-pass, Welch 10 s windows / 1 s hop, peak + SNR
src/app/cli.jl                   analyze_clip, command line
scripts/campulse.jl              offline heart rate from a clip
scripts/record_clip.jl           record a lossless clip (FFV1) + per-frame timestamps
scripts/camera_lock.ps1          lock / unlock exposure, white balance, focus
scripts/download_models.jl       fetch the YuNet model into data/models/
scripts/camera_check.jl          grab 150 frames, report size / fps / pixel format
scripts/list_camera_modes.jl     list DirectShow devices and the camera's modes
test/runtests.jl                 filters, resampling, ROI extraction, Welch estimator, end to end
```

## Setup and tests
Julia 1.11 (juliaup). From this folder:
```
julia --project -e "using Pkg; Pkg.instantiate(); Pkg.test()"
julia --project scripts/download_models.jl          # YuNet face model (data/models/, not in git)
```

## Measure heart rate from a clip
```
julia --project scripts/record_clip.jl 30 data/me.mkv
julia --project scripts/campulse.jl data/me.mkv --method green --csv data/me_hr.csv
julia --project scripts/campulse.jl data/me.mkv --roi 260,120,120,80   # fixed ROI, no face detector
```
Sit still, face the camera in steady light, about 50–80 cm away. `campulse` prints the
median bpm and SNR; the CSV has one row per second (`time_s,bpm,snr_db,valid_frac`).

## Camera
```
julia --project scripts/list_camera_modes.jl
julia --project scripts/camera_check.jl                      # C930e, 640x480 YUY2 @ 30
powershell -ExecutionPolicy Bypass -File scripts\camera_lock.ps1 status   # | lock | auto
```
Target mode: uncompressed YUY2 640x480 @ 30 fps (best colour fidelity). In dim light the
camera's auto-exposure lengthens the exposure and drops the frame rate (8 fps measured in a
dark room), so record in good light.

Auto exposure / white balance / focus wipe out the pulse, so `record_clip.jl` streams a few
seconds to let them settle, then locks them through DirectShow (`IAMCameraControl`,
`IAMVideoProcAmp`) and sets them back to auto after recording (`--keep-lock` to keep them).
Exposure is capped at -5 (1/32 s) so 30 fps is still possible. If the lock does not hold
(e.g. Logi Tune is running and resets it): open Logi Tune → camera → Image adjustments, turn
off auto exposure, auto white balance and auto focus, and set them by hand; or use the
DirectShow property page (`ffmpeg -f dshow -show_video_device_dialog true -i video="Logitech
Webcam C930e"`). `FrameMonitor` warns when mean brightness drifts >5 % within 2 s.

## Using the filters
```julia
using CamHealth.TemporalFilters
f  = ButterBandpass(PULSE, 30.0)          # 0.7–4 Hz, order 3
y  = apply(f, x)                          # zero-phase batch
st = init_state(f, first_frame)           # streaming, per pixel/element
yk = step!(st, frame_k)
```
