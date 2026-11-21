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

# Frame sources: (frame, timestamp) iterators
include("io/sources.jl")
export AbstractFrameSource, CameraSource, FileSource

# Pipeline stages (stubs, filled in from M1 on)
include("face/face.jl")
include("signal/rppg.jl")
include("hr/estimate.jl")
include("magnify/evm.jl")
include("app/app.jl")

end # module CamHealth
