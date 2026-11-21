# Frame sources: anything that yields `(frame, timestamp)` pairs.
#
#   for (frame, t) in CameraSource()   ... end
#   for (frame, t) in FileSource("clip.mp4") ... end
#
# `frame` is a Matrix{RGB{N0f8}} (VideoIO default); `t` is seconds (Float64) since
# the first frame. Real timestamps are kept so later stages can resample to a
# uniform rate and cope with dropped frames.

using VideoIO

"""Abstract supertype of frame sources. Iterating yields `(frame, timestamp_seconds)`."""
abstract type AbstractFrameSource end

Base.IteratorSize(::Type{<:AbstractFrameSource}) = Base.SizeUnknown()
Base.eltype(::Type{<:AbstractFrameSource}) = Tuple{Matrix{VideoIO.RGB{VideoIO.N0f8}},Float64}

"""
    CameraSource(device=nothing; kwargs...)

Live webcam. With no device the default camera is opened. On Windows (DirectShow)
the C930e is `"video=Logitech Webcam C930e"`; `kwargs` are passed to
`VideoIO.opencamera` (e.g. `options = (video_size="640x480", framerate="30")`).
Timestamps are wall-clock seconds measured when each frame is returned.
"""
mutable struct CameraSource <: AbstractFrameSource
    reader::Any
    t0::Float64
end

function CameraSource(device=nothing; kwargs...)
    reader = device === nothing ? VideoIO.opencamera(; kwargs...) :
             VideoIO.opencamera(device; kwargs...)
    CameraSource(reader, NaN)
end

function Base.iterate(src::CameraSource, state=nothing)
    eof(src.reader) && return nothing
    frame = read(src.reader)
    now = time()
    isnan(src.t0) && (src.t0 = now)
    return ((frame, now - src.t0), nothing)
end

"""
    FileSource(path)

Video file. Timestamps come from the stream's presentation times, relative to the
first frame.
"""
mutable struct FileSource <: AbstractFrameSource
    path::String
    reader::Any
    t0::Float64
end

FileSource(path::AbstractString) = FileSource(String(path), VideoIO.openvideo(path), NaN)

function Base.iterate(src::FileSource, state=nothing)
    eof(src.reader) && return nothing
    frame = read(src.reader)
    t = Float64(VideoIO.gettime(src.reader))
    isnan(src.t0) && (src.t0 = t)
    return ((frame, t - src.t0), nothing)
end

Base.close(src::Union{CameraSource,FileSource}) = close(src.reader)

"""Nominal frame rate of a file source (frames/s), from the container metadata."""
framerate(src::FileSource) = VideoIO.framerate(src.reader)
