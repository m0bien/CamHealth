"""
    CamHealth

Camera-based heart-rate monitoring (remote photoplethysmography, rPPG) with optional
Eulerian Video Magnification. One processing core, two frame sources (live camera,
video file). See PLAN.md for the roadmap.
"""
module CamHealth

# Standalone temporal filters (band-pass, EVM filters). Usable on their own:
#   using CamHealth.TemporalFilters
include("filters/TemporalFilters.jl")
using .TemporalFilters
export TemporalFilters, BandSpec, PULSE, BREATHING, ButterBandpass, DiffLowpass,
       apply, init_state, step!, ideal_bandpass

# Frame sources: (frame, timestamp) iterators, frame-interval / brightness monitoring
include("io/sources.jl")
export AbstractFrameSource, CameraSource, FileSource, FrameMonitor, lock_camera

# Face detection and skin ROI
include("face/face.jl")
export Face, YuNetDetector, detect, FaceTracker, FixedROI, update!, face_roi, skin_mask, roi_rects

# Signal extraction and rPPG methods
include("signal/extract.jl")
include("signal/rppg.jl")
export RGBTrace, extract_trace, resample_uniform, rppg_green

# Heart-rate estimation
include("hr/estimate.jl")
export HRTrack, estimate_hr, spectrum_peak, welch_psd

# Pipelines / CLI
include("magnify/evm.jl")
include("app/app.jl")
include("app/cli.jl")
export analyze_clip

end # module CamHealth
