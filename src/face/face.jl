# Face detection (YuNet via ONNX Runtime), landmark-based skin ROIs, tracking.
#
#   det = YuNetDetector()                      # loads data/models/face_detection_yunet_2023mar.onnx
#   faces = detect(det, frame)                 # Vector{Face}, best first
#   tracker = FaceTracker(det; every=10)
#   roi = update!(tracker, frame)              # BitMatrix skin ROI (or nothing if no face yet)
#
# Fallback without a detector: FixedROI(x, y, w, h) gives a constant rectangular ROI.

import ONNXRunTime as ORT
using Colors: RGB, red, green, blue

const YUNET_FILE = "face_detection_yunet_2023mar.onnx"
default_model_path() = joinpath(dirname(dirname(@__DIR__)), "data", "models", YUNET_FILE)

"""A detected face: box `(x, y, w, h)` in pixels (x = column, y = row, 0-based top-left
corner), detection `score`, and 5 landmarks as a 2x5 matrix of (x, y):
right eye, left eye, nose tip, right mouth corner, left mouth corner (subject's
right/left, i.e. image left first)."""
struct Face
    box::NTuple{4,Float64}
    score::Float64
    landmarks::Matrix{Float64}   # 2 x 5
end

# ---------------------------------------------------------------------------
# YuNet
# ---------------------------------------------------------------------------

struct YuNetDetector
    session::Any
    insize::Int          # model input is insize x insize (640 for the 2023mar export)
    score_threshold::Float64
    nms_threshold::Float64
end

"""
    YuNetDetector(path=default_model_path(); score_threshold=0.8, nms_threshold=0.3)

Load OpenCV Zoo's YuNet (2023mar) ONNX model with ONNX Runtime (CPU). Download it with
`julia --project scripts/download_models.jl`.
"""
function YuNetDetector(path::AbstractString=default_model_path(); score_threshold=0.8, nms_threshold=0.3)
    isfile(path) || error("YuNet model not found at $path — run scripts/download_models.jl")
    YuNetDetector(ORT.load_inference(path), 640, score_threshold, nms_threshold)
end

"""Scale factor and BGR float tensor (1x3xSxS, 0–255, zero padded bottom/right)."""
function yunet_input(frame::AbstractMatrix{<:RGB}, S::Integer)
    H, W = size(frame)
    s = min(S / H, S / W, 1.0)
    img = s < 1 ? _resize_nn(frame, round(Int, H * s), round(Int, W * s)) : frame
    h, w = size(img)
    x = zeros(Float32, 1, 3, S, S)
    @inbounds for j in 1:w, i in 1:h
        p = img[i, j]
        x[1, 1, i, j] = 255f0 * Float32(blue(p))
        x[1, 2, i, j] = 255f0 * Float32(green(p))
        x[1, 3, i, j] = 255f0 * Float32(red(p))
    end
    return x, s
end

function _resize_nn(img, h, w)
    H, W = size(img)
    [img[clamp(round(Int, (i - 0.5) * H / h + 0.5), 1, H), clamp(round(Int, (j - 0.5) * W / w + 0.5), 1, W)] for i in 1:h, j in 1:w]
end

"""
    detect(det::YuNetDetector, frame) -> Vector{Face}

Run YuNet on an RGB frame. Faces are returned sorted by score (best first), with
coordinates in the original frame's pixels.
"""
function detect(det::YuNetDetector, frame::AbstractMatrix{<:RGB})
    x, s = yunet_input(frame, det.insize)
    out = det.session(Dict("input" => x))
    faces = yunet_decode(out, det.insize, det.score_threshold)
    faces = nms(faces, det.nms_threshold)
    # back to original resolution
    return [Face(f.box ./ s, f.score, f.landmarks ./ s) for f in faces]
end

"""Decode YuNet's per-stride outputs (cls, obj, bbox, kps) into faces (OpenCV's
FaceDetectorYN post-processing)."""
function yunet_decode(out, S::Integer, score_threshold::Real)
    faces = Face[]
    for stride in (8, 16, 32)
        cls = out["cls_$stride"]; obj = out["obj_$stride"]
        bb = out["bbox_$stride"]; kp = out["kps_$stride"]
        cols = S ÷ stride
        n = size(cls, 2)
        @inbounds for k in 1:n
            score = sqrt(clamp(cls[1, k, 1], 0f0, 1f0) * clamp(obj[1, k, 1], 0f0, 1f0))
            score < score_threshold && continue
            r, c = divrem(k - 1, cols)
            cx = (c + bb[1, k, 1]) * stride
            cy = (r + bb[1, k, 2]) * stride
            w = exp(bb[1, k, 3]) * stride
            h = exp(bb[1, k, 4]) * stride
            lm = Matrix{Float64}(undef, 2, 5)
            for m in 1:5
                lm[1, m] = (kp[1, k, 2m-1] + c) * stride
                lm[2, m] = (kp[1, k, 2m] + r) * stride
            end
            push!(faces, Face((cx - w / 2, cy - h / 2, w, h), score, lm))
        end
    end
    return faces
end

function iou(a::NTuple{4,Float64}, b::NTuple{4,Float64})
    x1 = max(a[1], b[1]); y1 = max(a[2], b[2])
    x2 = min(a[1] + a[3], b[1] + b[3]); y2 = min(a[2] + a[4], b[2] + b[4])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inter / (a[3] * a[4] + b[3] * b[4] - inter + eps())
end

function nms(faces::Vector{Face}, thr::Real)
    order = sort(faces; by=f -> -f.score)
    keep = Face[]
    for f in order
        all(k -> iou(k.box, f.box) <= thr, keep) && push!(keep, f)
    end
    return keep
end

# ---------------------------------------------------------------------------
# ROI from landmarks + skin mask
# ---------------------------------------------------------------------------

"""
    roi_rects(face) -> Vector{NTuple{4,Float64}}

Forehead and both cheek rectangles `(x, y, w, h)` derived from the 5 landmarks,
scaled by the inter-ocular distance `d`.
"""
function roi_rects(f::Face)
    re, le = f.landmarks[:, 1], f.landmarks[:, 2]
    rm, lm = f.landmarks[:, 4], f.landmarks[:, 5]
    d = hypot((le .- re)...)
    mid = (re .+ le) ./ 2
    forehead = (mid[1] - 0.45d, mid[2] - 0.75d, 0.9d, 0.35d)
    cheek(eye, mouth) = begin
        c = (eye[1] * 0.6 + mouth[1] * 0.4, eye[2] * 0.4 + mouth[2] * 0.6)
        (c[1] - 0.2d, c[2] - 0.2d, 0.4d, 0.4d)
    end
    return [forehead, cheek(re, rm), cheek(le, lm)]
end

"""Boolean mask of plausible skin pixels (YCbCr thresholds, Chai & Ngan 1999)."""
function skin_mask(frame::AbstractMatrix{<:RGB})
    m = falses(size(frame))
    @inbounds for i in eachindex(frame)
        p = frame[i]
        r, g, b = 255 * Float64(red(p)), 255 * Float64(green(p)), 255 * Float64(blue(p))
        cb = 128 - 0.168736r - 0.331264g + 0.5b
        cr = 128 + 0.5r - 0.418688g - 0.081312b
        y = 0.299r + 0.587g + 0.114b
        m[i] = (77 <= cb <= 127) & (133 <= cr <= 173) & (y > 30) & (y < 250)
    end
    return m
end

"""Mask with `true` inside the given rectangles (clipped to `sz`)."""
function rect_mask(sz::Tuple{Int,Int}, rects)
    m = falses(sz)
    for (x, y, w, h) in rects
        r1 = clamp(floor(Int, y) + 1, 1, sz[1]); r2 = clamp(floor(Int, y + h), 1, sz[1])
        c1 = clamp(floor(Int, x) + 1, 1, sz[2]); c2 = clamp(floor(Int, x + w), 1, sz[2])
        m[r1:r2, c1:c2] .= true
    end
    return m
end

"""ROI mask for a face: landmark rectangles ∩ skin pixels. If the skin test removes
almost everything (unusual lighting), the plain rectangles are used."""
function face_roi(frame::AbstractMatrix{<:RGB}, f::Face; use_skin=true)
    rm = rect_mask(size(frame), roi_rects(f))
    use_skin || return rm
    m = rm .& skin_mask(frame)
    return count(m) >= 0.2 * count(rm) ? m : rm
end

# ---------------------------------------------------------------------------
# Tracking: detect every N frames, EMA-smooth box + landmarks
# ---------------------------------------------------------------------------

"""Anything that turns a frame into an ROI mask (or `nothing` when unavailable)."""
abstract type AbstractROI end

"""Constant rectangle ROI `(x, y, w, h)` in pixels (0-based corner) — manual fallback."""
struct FixedROI <: AbstractROI
    rect::NTuple{4,Float64}
end
FixedROI(x, y, w, h) = FixedROI(Float64.((x, y, w, h)))
update!(r::FixedROI, frame) = rect_mask(size(frame), [r.rect])

mutable struct FaceTracker <: AbstractROI
    det::YuNetDetector
    every::Int
    alpha::Float64            # EMA weight of a new detection
    use_skin::Bool
    n::Int
    face::Union{Nothing,Face} # smoothed face
    lost::Int                 # consecutive detections without a face
end

"""
    FaceTracker(det; every=10, alpha=0.5, use_skin=true)

Runs the detector every `every` frames, smooths box and landmarks with an exponential
moving average, and returns the ROI mask for every frame. After 3 failed detections in
a row the face is dropped and `update!` returns `nothing`.
"""
FaceTracker(det::YuNetDetector; every=10, alpha=0.5, use_skin=true) =
    FaceTracker(det, every, alpha, use_skin, 0, nothing, 0)

function update!(t::FaceTracker, frame::AbstractMatrix{<:RGB})
    if t.n % t.every == 0 || t.face === nothing
        faces = detect(t.det, frame)
        if isempty(faces)
            t.lost += 1
            t.lost >= 3 && (t.face = nothing)
        else
            t.lost = 0
            f = faces[1]
            if t.face === nothing
                t.face = f
            else
                a = t.alpha; o = t.face
                t.face = Face(a .* f.box .+ (1 - a) .* o.box, f.score, a .* f.landmarks .+ (1 - a) .* o.landmarks)
            end
        end
    end
    t.n += 1
    t.face === nothing && return nothing
    return face_roi(frame, t.face; use_skin=t.use_skin)
end
