# CamHealth: plan for measuring heart rate from face colour (Julia)

Goal: measure heart rate from the small colour changes that blood flow causes in the skin of the face. This is called remote photoplethysmography (rPPG). It should work in two ways:
- **Live**, from the Logitech C930e webcam.
- **Offline**, from a video file.

Both share one processing core.

## Hardware and constraints (checked 2026-11-20)
- Camera: Logitech C930e (USB 2.0, VID 046D, PID 0843). Logitech's published specs give at most 1080p at 30 fps, with MJPEG, YUY2 and H.264 formats. These still need to be confirmed on the camera with ffmpeg.
  - 30 fps is enough. Heart rate lies between 0.7 and 4 Hz (42–240 bpm), well under the 15 Hz limit that 30 fps allows.
  - Choose the capture format for colour fidelity, not resolution. Uncompressed YUY2 at 640x480 and 30 fps is the best choice. MJPEG at 720p and 30 fps is the fallback, because compression adds colour noise.
  - **Lock exposure, white balance and focus.** If the camera adjusts these automatically, it wipes out the pulse signal.
- PC: Intel Core i9-12900HK with Intel Iris Xe graphics. The CPU alone is plenty, so no GPU is needed.
- Julia 1.11.6 is installed. ffmpeg still needs to be installed.

## Pipeline (one core, two frame sources)
```
Frame source (VideoFile | Camera)
  -> face detection (every N frames) + tracking
  -> skin region: forehead + both cheeks (skin mask)
  -> average R, G, B over the region for each frame  -> time series with timestamps
  -> resample to a uniform rate (uses real timestamps, so dropped frames are handled)
  -> POS algorithm (Wang et al. 2017); CHROM and green-channel kept as alternatives
  -> band-pass filter 0.7–4 Hz (Butterworth)
  -> heart rate: Welch power spectrum peak, refined by interpolating around the peak
     + a check on the time between beats (optional)
  -> signal quality score (SNR) and flags for motion or lighting changes
  -> output: bpm over time, the pulse waveform, quality score
```
Live mode uses a sliding window of 8–10 s and updates the estimate every 0.5–1 s.
Video mode processes the whole clip, then exports the bpm curve to CSV and saves plots.

**MIT Eulerian Video Magnification** (Wu et al. 2012): an optional visual mode. It applies a band-pass filter over time to a Gaussian or Laplacian pyramid of each frame, then amplifies the colour changes. This makes the pulse visible as the face turning red in rhythm. It is not needed to measure heart rate; it is for showing and demonstrating the signal.

## Julia packages
- Video and camera input: VideoIO.jl (uses ffmpeg; webcam via DirectShow on Windows)
- Face detection: OpenCV.jl, using the YuNet face detector, falling back to a Haar cascade
  - If landmark-accurate regions are needed later: MediaPipe Face Mesh via PythonCall.jl
- Signal processing: DSP.jl, FFTW.jl, Interpolations.jl, StatsBase.jl
- Live view and plots: GLMakie.jl, showing the video with the region overlay, the pulse trace and the bpm
- Command-line interface: Comonicon.jl or ArgParse.jl
- Startup speed: PrecompileTools.jl, plus a sysimage built with PackageCompiler.jl so it doesn't pause on first run

## Project layout
```
CamHealth/
  Project.toml
  src/CamHealth.jl
  src/sources.jl      # VideoFileSource, CameraSource, ring buffer
  src/face.jl         # detect/track, region masks
  src/rppg.jl         # POS, CHROM, GREEN
  src/estimate.jl     # filters, Welch, bpm, SNR
  src/evm.jl          # Eulerian magnification (optional)
  src/ui.jl           # GLMakie live dashboard
  bin/camhealth.jl    # `camhealth live` / `camhealth file clip.mp4`
  test/               # synthetic-signal tests + dataset checks
```

## Phases
0. **Setup.** Install ffmpeg and confirm the camera's real modes. Create the Julia project and install the packages. Confirm VideoIO can open the C930e, and work out how to lock its exposure.
1. **Offline core.** Run the full pipeline on video files. Write tests against synthetic signals with a known heart rate.
2. **Validation.** Test against the public UBFC-rPPG dataset, which includes reference measurements. Also record your own clips while wearing a fingertip pulse oximeter. Target: mean error under 3 bpm when sitting still.
3. **Live mode.** Run capture and processing on separate threads, plus the GLMakie dashboard. Target: under 1 s of delay, with about 30 fps processed in real time.
4. **Robustness.** Handle motion (region tracking, rejecting bad windows), lighting flicker (filtering out 50/100 Hz light flicker) and different skin tones (POS helps here), and show a confidence score.
5. **EVM visualisation** and exporting magnified video (optional).
6. **Later:** breathing rate from the same signal (0.1–0.5 Hz) and heart-rate variability.

## Risks
- Face detection support in Julia is the weakest part. OpenCV.jl is the main plan and PythonCall with MediaPipe is the fallback.
- Automatic camera settings and MJPEG compression add noise, so both need locking down early.
- The slow first run of Julia code is handled with PrecompileTools and a sysimage.
