# CamHealth

Camera-based heart-rate monitoring in Julia: remote photoplethysmography (rPPG) from the
tiny colour changes blood flow causes in facial skin, plus optional MIT Eulerian Video
Magnification (EVM) to make the pulse visible. Works live from a webcam (Logitech C930e)
or offline from a video file. See [PLAN.md](PLAN.md) for the full roadmap.

**Status: M2 (accurate rPPG + HRV)** — POS (default), CHROM and green methods, SNR quality
gate, beat detection and HRV (RMSSD, SDNN, pNN50), benchmark harness for UBFC-rPPG / PURE.
M1 delivered lossless recording with locked camera settings, YuNet face ROI and the CLI.

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
src/signal/rppg.jl               green, CHROM (de Haan 2013), POS (Wang 2017); PPG-like sign
src/hr/estimate.jl               detrend, band-pass, Welch 10 s windows / 1 s hop, peak + SNR,
                                 quality gate (SNR >= -3 dB, jumps <= 15 bpm/s)
src/hr/beats.jl                  beats (250 Hz spline, peak + foot), IBI cleaning, RMSSD/SDNN/pNN50
src/bench/                       UBFC-rPPG / PURE loaders, synthetic dataset, metrics, run_benchmark
src/app/cli.jl                   analyze_clip, command line
scripts/campulse.jl              offline heart rate (and HRV) from a clip
scripts/benchmark.jl             rPPG vs contact PPG: MAE, RMSE, r, Bland-Altman, IBI / RMSSD error
scripts/record_clip.jl           record a lossless clip (FFV1) + per-frame timestamps
scripts/camera_lock.ps1          lock / unlock exposure, white balance, focus
scripts/download_models.jl       fetch the YuNet model into data/models/
scripts/camera_check.jl          grab 150 frames, report size / fps / pixel format
scripts/list_camera_modes.jl     list DirectShow devices and the camera's modes
test/runtests.jl                 filters, ROI, methods under disturbances, gate, beats, HRV, loaders
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
julia --project scripts/campulse.jl data/me.mkv                       # POS, quality-gated
julia --project scripts/campulse.jl data/me.mkv --method all --hrv --csv data/me_hr.csv
julia --project scripts/campulse.jl data/me.mkv --roi 260,120,120,80   # fixed ROI, no face detector
```
Sit still, face the camera in steady light, about 50–80 cm away. `campulse` prints the
median bpm and SNR over windows that pass the quality gate; the CSV has one row per second
(`time_s,bpm,snr_db,valid_frac,quality`, bpm left empty for "low quality" windows). `--hrv`
writes `data/<clip>_beats.csv` (peak and foot times, IBIs) and prints SDNN / RMSSD / pNN50.
HRV needs a clean signal: at SNR around 0 dB the beat timing jitter dominates RMSSD.

## Benchmark
```
julia --project scripts/benchmark.jl synthetic                 # harness check on generated clips
julia --project scripts/benchmark.jl ubfc2 data/datasets/ubfc2   # UBFC-rPPG dataset 2
julia --project scripts/benchmark.jl pure  data/datasets/pure    # PURE
```
Results: `results/benchmark_<dataset>.csv` and `results/README.md`. The datasets are not
redistributable and stay in `data/datasets/` (git-ignored). UBFC-rPPG: folder links on
https://sites.google.com/view/ybenezeth/ubfcrppg (research use, cite Bobbia et al. 2017).
PURE: request access from TU Ilmenau by email.

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
