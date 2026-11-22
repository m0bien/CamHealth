# Frame sources: anything that yields `(frame, timestamp)` pairs.
#
#   for (frame, t) in CameraSource()   ... end
#   for (frame, t) in FileSource("clip.mkv") ... end
#
# `frame` is a Matrix{RGB{N0f8}} (VideoIO default); `t` is seconds (Float64) since
# the first frame. Real timestamps are kept so later stages can resample to a
# uniform rate and cope with dropped frames.

using VideoIO
using Statistics: mean
using Colors: red, green, blue

"""Abstract supertype of frame sources. Iterating yields `(frame, timestamp_seconds)`."""
abstract type AbstractFrameSource end

Base.IteratorSize(::Type{<:AbstractFrameSource}) = Base.SizeUnknown()
Base.eltype(::Type{<:AbstractFrameSource}) = Tuple{Matrix{VideoIO.RGB{VideoIO.N0f8}},Float64}

# ---------------------------------------------------------------------------
# Frame monitor: intervals, drops, brightness drift
# ---------------------------------------------------------------------------

"""
    FrameMonitor(; nominal_fps=30, drop_factor=1.5, drift_window=2.0, drift_tol=0.05)

Tracks frame intervals and mean brightness. An interval longer than
`drop_factor / nominal_fps` counts as a drop (`ndropped` adds the estimated number of
missing frames). A brightness change of more than `drift_tol` (relative) within
`drift_window` seconds is counted as a drift event and warned about once per window —
usually auto-exposure / auto white balance still running, or lighting changes.
"""
mutable struct FrameMonitor
    nominal_fps::Float64
    drop_factor::Float64
    drift_window::Float64
    drift_tol::Float64
    warn::Bool
    times::Vector{Float64}
    brightness::Vector{Float64}
    ndropped::Int
    ndrop_events::Int
    ndrift::Int
    last_drift_warn::Float64
end

FrameMonitor(; nominal_fps=30.0, drop_factor=1.5, drift_window=2.0, drift_tol=0.05, warn=true) =
    FrameMonitor(nominal_fps, drop_factor, drift_window, drift_tol, warn, Float64[], Float64[], 0, 0, 0, -Inf)

"""Mean brightness of a frame (average of R, G, B on a sparse grid; cheap)."""
function frame_brightness(frame::AbstractMatrix)
    s = 0.0; n = 0
    @inbounds for j in 1:4:size(frame, 2), i in 1:4:size(frame, 1)
        p = frame[i, j]
        s += (Float64(red(p)) + Float64(green(p)) + Float64(blue(p))) / 3
        n += 1
    end
    s / max(n, 1)
end

"""Record one frame at time `t`. Returns `true` if a drop was detected before it."""
function record!(m::FrameMonitor, frame, t::Real)
    dropped = false
    if !isempty(m.times)
        dt = t - m.times[end]
        nominal = 1 / m.nominal_fps
        if dt > m.drop_factor * nominal
            m.ndrop_events += 1
            m.ndropped += max(1, round(Int, dt / nominal) - 1)
            dropped = true
        end
    end
    b = frame_brightness(frame)
    push!(m.times, t); push!(m.brightness, b)
    # brightness drift within the window
    k = searchsortedfirst(m.times, t - m.drift_window)
    if t - m.times[k] >= 0.9 * m.drift_window
        ref = m.brightness[k]
        if ref > 0 && abs(b - ref) / ref > m.drift_tol
            m.ndrift += 1
            if m.warn && t - m.last_drift_warn > m.drift_window
                @warn "brightness drifted $(round(100 * (b - ref) / ref, digits=1))% in $(m.drift_window) s — auto exposure/white balance active, or lighting changed?" t
                m.last_drift_warn = t
            end
        end
    end
    return dropped
end

intervals(m::FrameMonitor) = diff(m.times)

function Base.show(io::IO, m::FrameMonitor)
    n = length(m.times)
    fps = n > 1 ? (n - 1) / (m.times[end] - m.times[1]) : NaN
    print(io, "FrameMonitor($n frames, $(round(fps, digits=2)) fps, $(m.ndropped) dropped in $(m.ndrop_events) gaps, $(m.ndrift) drift samples)")
end

# ---------------------------------------------------------------------------
# Camera lock (Windows DirectShow, via scripts/camera_lock.ps1)
# ---------------------------------------------------------------------------

const CAMERA_LOCK_SCRIPT = joinpath(dirname(dirname(@__DIR__)), "scripts", "camera_lock.ps1")

"""
    lock_camera(action=:lock; device="Logitech Webcam C930e", exposure=nothing, whitebalance=nothing, focus=nothing)

Lock (`:lock`), unlock (`:auto`) or show (`:status`) exposure, white balance and focus
using DirectShow's IAMCameraControl / IAMVideoProcAmp (Windows only). `:lock` freezes
the values auto mode has currently chosen, so call it while the camera is streaming
under the lighting you will record in. Returns the script output.
"""
function lock_camera(action::Symbol=:lock; device="Logitech Webcam C930e",
                     exposure=nothing, whitebalance=nothing, focus=nothing)
    Sys.iswindows() || error("lock_camera uses DirectShow and only works on Windows")
    args = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", CAMERA_LOCK_SCRIPT, String(action), "-Device", device]
    exposure === nothing || append!(args, ["-Exposure", string(exposure)])
    whitebalance === nothing || append!(args, ["-WhiteBalance", string(whitebalance)])
    focus === nothing || append!(args, ["-Focus", string(focus)])
    return read(`powershell $args`, String)
end

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

"""
    CameraSource(device="video=Logitech Webcam C930e"; video_size="640x480", framerate=30,
                 pixel_format="yuyv422", monitor=FrameMonitor())

Live webcam through FFmpeg's DirectShow input (Windows). The default mode is the C930e's
uncompressed YUY2 640x480 @ 30 fps. Timestamps are wall-clock seconds measured when each
frame is returned; every frame is recorded in `src.monitor` (intervals, drops,
brightness drift). Use `lock_camera()` to freeze exposure / white balance / focus.
"""
mutable struct CameraSource <: AbstractFrameSource
    reader::Any
    t0::Float64
    monitor::FrameMonitor
end

function CameraSource(device::AbstractString="video=Logitech Webcam C930e"; video_size="640x480",
                      framerate=30, pixel_format="yuyv422", monitor=FrameMonitor(nominal_fps=framerate))
    fmt = VideoIO.libffmpeg.av_find_input_format(Sys.iswindows() ? "dshow" : "video4linux2")
    opts = VideoIO.AVDict("video_size" => video_size, "framerate" => string(framerate))
    isempty(pixel_format) || (opts["pixel_format"] = pixel_format)
    CameraSource(VideoIO.opencamera(device, fmt, opts), NaN, monitor)
end

function Base.iterate(src::CameraSource, state=nothing)
    eof(src.reader) && return nothing
    frame = read(src.reader)
    now = time()
    isnan(src.t0) && (src.t0 = now)
    t = now - src.t0
    record!(src.monitor, frame, t)
    return ((frame, t), nothing)
end

"""
    FileSource(path; monitor=nothing)

Video file. Timestamps come from the stream's presentation times, relative to the
first frame. Pass a `FrameMonitor` to collect interval / drop statistics.
"""
mutable struct FileSource <: AbstractFrameSource
    path::String
    reader::Any
    t0::Float64
    monitor::Union{Nothing,FrameMonitor}
end

FileSource(path::AbstractString; monitor=nothing) =
    FileSource(String(path), VideoIO.openvideo(path), NaN, monitor)

function Base.iterate(src::FileSource, state=nothing)
    eof(src.reader) && return nothing
    frame = read(src.reader)
    t = Float64(VideoIO.gettime(src.reader))
    isnan(src.t0) && (src.t0 = t)
    t -= src.t0
    src.monitor === nothing || record!(src.monitor, frame, t)
    return ((frame, t), nothing)
end

Base.close(src::Union{CameraSource,FileSource}) = close(src.reader)

"""Nominal frame rate of a file source (frames/s), from the container metadata."""
framerate(src::FileSource) = VideoIO.framerate(src.reader)
