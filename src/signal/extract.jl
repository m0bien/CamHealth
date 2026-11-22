# Frame sequence -> ROI colour trace -> uniformly resampled trace.

using Interpolations: linear_interpolation, Flat
using Statistics: mean, median

"""
Raw colour trace: per-frame timestamp `t` (s), mean `rgb` over the ROI (N x 3, columns
R, G, B in 0..1), `valid` (false where no face / ROI was available) and `npix` (ROI size).
"""
struct RGBTrace
    t::Vector{Float64}
    rgb::Matrix{Float64}
    valid::BitVector
    npix::Vector{Int}
end

Base.length(tr::RGBTrace) = length(tr.t)

"""Mean (R, G, B) of `frame` over `mask`, and the number of pixels."""
function roi_mean(frame::AbstractMatrix, mask::AbstractMatrix{Bool})
    size(frame) == size(mask) || throw(DimensionMismatch("frame $(size(frame)) vs mask $(size(mask))"))
    r = g = b = 0.0; n = 0
    @inbounds for i in eachindex(frame, mask)
        if mask[i]
            p = frame[i]
            r += Float64(red(p)); g += Float64(green(p)); b += Float64(blue(p)); n += 1
        end
    end
    n == 0 && return (NaN, NaN, NaN), 0
    return (r / n, g / n, b / n), n
end

"""
    extract_trace(source, roi; maxframes=typemax(Int), min_pixels=50, verbose=false) -> RGBTrace

Iterate a frame source, get the ROI mask per frame (`update!(roi, frame)`), and average
the colour over it. Frames without a face (mask `nothing` or fewer than `min_pixels`
pixels) are marked invalid.
"""
function extract_trace(src, roi; maxframes::Integer=typemax(Int), min_pixels::Integer=50, verbose=false)
    ts = Float64[]; rows = NTuple{3,Float64}[]; valid = Bool[]; npix = Int[]
    for (k, (frame, t)) in enumerate(src)
        k > maxframes && break
        mask = update!(roi, frame)
        if mask === nothing
            c, n = (NaN, NaN, NaN), 0
        else
            c, n = roi_mean(frame, mask)
        end
        ok = n >= min_pixels
        push!(ts, t); push!(rows, ok ? c : (NaN, NaN, NaN)); push!(valid, ok); push!(npix, n)
        verbose && k % 150 == 0 && println("  frame $k  t=$(round(t, digits=1)) s  roi=$(n) px  face=$(ok)")
    end
    rgb = isempty(rows) ? zeros(0, 3) : [rows[i][j] for i in eachindex(rows), j in 1:3]
    return RGBTrace(ts, rgb, BitVector(valid), npix)
end

"""
    resample_uniform(t, X, fs; valid=trues(length(t)), max_gap=0.25) -> (tu, Xu, validu)

Linearly interpolate the samples `X` (vector or N x k matrix) taken at times `t` onto a
uniform grid `t[1]:1/fs:t[end]`, using only `valid` samples. A grid point is valid when
the valid samples around it are no more than `max_gap` seconds apart (short dropouts,
e.g. dropped frames, are bridged; long face losses stay marked invalid).
"""
function resample_uniform(t::AbstractVector{<:Real}, X::AbstractVecOrMat, fs::Real;
                          valid::AbstractVector{Bool}=trues(length(t)), max_gap::Real=0.25)
    length(t) == size(X, 1) || throw(DimensionMismatch("t and X differ in length"))
    tv = Float64.(t[valid]); Xv = X[valid, :]
    # strictly increasing knots (drop duplicate timestamps)
    keep = [true; diff(tv) .> 0]
    tv = tv[keep]; Xv = Xv[keep, :]
    length(tv) >= 2 || error("need at least two valid samples to resample")
    tu = collect(t[1]:1/fs:t[end])
    Xu = similar(Xv, length(tu), size(Xv, 2))
    for j in axes(Xv, 2)
        itp = linear_interpolation(tv, Xv[:, j]; extrapolation_bc=Flat())
        Xu[:, j] .= itp.(tu)
    end
    validu = falses(length(tu))
    for (k, τ) in enumerate(tu)
        i = searchsortedlast(tv, τ)            # last knot <= τ
        if 1 <= i < length(tv)
            validu[k] = tv[i+1] - tv[i] <= max_gap
        elseif i == length(tv)
            validu[k] = τ - tv[end] <= max_gap
        else
            validu[k] = tv[1] - τ <= max_gap
        end
    end
    return tu, (X isa AbstractVector ? vec(Xu) : Xu), validu
end

resample_uniform(tr::RGBTrace, fs::Real; kw...) = resample_uniform(tr.t, tr.rgb, fs; valid=tr.valid, kw...)
