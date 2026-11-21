# CamHealth

Camera-based heart-rate monitoring in Julia: remote photoplethysmography (rPPG) from the
tiny colour changes blood flow causes in facial skin, plus optional MIT Eulerian Video
Magnification (EVM) to make the pulse visible. Works live from a webcam (Logitech C930e)
or offline from a video file. See [PLAN.md](PLAN.md) for the full roadmap.

**Status: M0** — package skeleton, `TemporalFilters`, synthetic tests, camera check.

## Layout
```
src/CamHealth.jl                 main module
src/io/sources.jl                AbstractFrameSource, CameraSource, FileSource -> (frame, t)
src/filters/TemporalFilters.jl   standalone: BandSpec, PULSE, BREATHING, Butterworth band-pass
                                 (batch filtfilt + streaming step!), DiffLowpass (EVM), ideal FFT band-pass
src/face, signal, hr, magnify, app   stubs (M1+)
scripts/camera_check.jl          open the webcam, grab 150 frames, report size / fps / pixel format
scripts/list_camera_modes.jl     list DirectShow devices and the camera's modes (bundled ffmpeg)
test/runtests.jl                 filter response, streaming vs batch, synthetic 72 bpm rPPG
```

## Setup and tests
Julia 1.11 (juliaup). From this folder:
```
julia --project -e "using Pkg; Pkg.instantiate(); Pkg.test()"
```

## Camera
```
julia --project scripts/list_camera_modes.jl
julia --project scripts/camera_check.jl                      # C930e, 640x480 YUY2 @ 30
julia --project scripts/camera_check.jl "video=<device>" 640x480
```
Target mode: uncompressed YUY2 640x480 @ 30 fps (best colour fidelity); MJPEG 720p30 is the
fallback. In dim light the camera's auto-exposure drops the frame rate (8 fps was measured
on the first run), so add light and lock exposure / white balance / focus (Logitech
software or the DirectShow property page, `-show_video_device_dialog true` in ffmpeg).

## Using the filters
```julia
using CamHealth.TemporalFilters
f  = ButterBandpass(PULSE, 30.0)          # 0.7–4 Hz, order 3
y  = apply(f, x)                          # zero-phase batch
st = init_state(f, first_frame)           # streaming, per pixel/element
yk = step!(st, frame_k)
```
