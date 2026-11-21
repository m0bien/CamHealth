"""
    TemporalFilters

Standalone temporal (time-axis) filters for rPPG and Eulerian Video Magnification.

* `BandSpec`, `PULSE` (0.7–4 Hz), `BREATHING` (0.1–0.5 Hz)
* `ButterBandpass`: Butterworth band-pass (second-order sections via DSP.jl)
    - batch:     `apply(f, x)` — zero-phase `filtfilt` along `dims` (default 1)
    - streaming: `st = init_state(f, x0); y = step!(st, x)` — causal, one sample at a
      time; `x` may be a scalar or an array (filtered per element, e.g. a frame)
* `DiffLowpass`: difference of two first-order low-passes (per-pixel IIR band-pass used
  by Wu et al. 2012 EVM), streaming via `init_state`/`step!`
* `ideal_bandpass`: ideal FFT band-pass (batch, zero out-of-band bins)

This module only depends on DSP and FFTW so it can be used on its own.
"""
module TemporalFilters

using DSP
using FFTW

export BandSpec, PULSE, BREATHING
export ButterBandpass, DiffLowpass
export apply, init_state, step!, ideal_bandpass

# ---------------------------------------------------------------------------
# Band specification
# ---------------------------------------------------------------------------

"""Pass band in Hz: `lo` < `hi`."""
struct BandSpec
    lo::Float64
    hi::Float64
    function BandSpec(lo::Real, hi::Real)
        0 <= lo < hi || throw(ArgumentError("need 0 <= lo < hi, got lo=$lo hi=$hi"))
        new(Float64(lo), Float64(hi))
    end
end

"""Heart-rate band, 0.7–4 Hz (42–240 bpm)."""
const PULSE = BandSpec(0.7, 4.0)
"""Breathing band, 0.1–0.5 Hz (6–30 breaths/min)."""
const BREATHING = BandSpec(0.1, 0.5)

function _check_band(band::BandSpec, fs::Real)
    band.lo > 0 || throw(ArgumentError("band-pass needs lo > 0"))
    band.hi < fs / 2 || throw(ArgumentError("band hi=$(band.hi) Hz must be below Nyquist $(fs/2) Hz"))
end

# ---------------------------------------------------------------------------
# Butterworth band-pass
# ---------------------------------------------------------------------------

"""
    ButterBandpass(band::BandSpec, fs; order=3)

Butterworth band-pass of the given `order` (prototype order; the band-pass has
`2*order` poles). Stored as second-order sections for numerical stability.
"""
struct ButterBandpass
    band::BandSpec
    fs::Float64
    order::Int
    sos::SecondOrderSections{:z,Float64,Float64}
end

function ButterBandpass(band::BandSpec, fs::Real; order::Integer=3)
    _check_band(band, fs)
    zpk = digitalfilter(Bandpass(band.lo, band.hi), Butterworth(order); fs=fs)
    sos = convert(SecondOrderSections, zpk)
    ButterBandpass(band, Float64(fs), Int(order), sos)
end

"""
    apply(f::ButterBandpass, x::AbstractArray; dims=1)

Zero-phase (forward-backward, `filtfilt`) band-pass of `x` along `dims`.
"""
apply(f::ButterBandpass, x::AbstractVector{<:Real}) = filtfilt(f.sos, float.(x))
function apply(f::ButterBandpass, x::AbstractArray{<:Real}; dims::Integer=1)
    mapslices(v -> filtfilt(f.sos, float.(v)), x; dims=dims)
end

"""
    apply_causal(f::ButterBandpass, x::AbstractVector)

One-directional (causal) filtering of a whole vector — identical to what the
streaming `step!` produces sample by sample. Mainly for testing.
"""
apply_causal(f::ButterBandpass, x::AbstractVector{<:Real}) = filt(f.sos, float.(x))

# Streaming state: Direct-Form II transposed, one (z1, z2) pair per biquad.
# Each z is either a scalar or an array shaped like the input sample.
mutable struct ButterState{T}
    b0::Vector{Float64}; b1::Vector{Float64}; b2::Vector{Float64}
    a1::Vector{Float64}; a2::Vector{Float64}
    gain::Float64
    z1::Vector{T}
    z2::Vector{T}
    buf::T            # work buffer (arrays only; unused for scalars)
end

"""
    init_state(f, x0)

Create a zero-initialised streaming state for filter `f`. `x0` is a template sample
(a scalar, or an array such as a frame) that fixes element type and shape.
"""
function init_state(f::ButterBandpass, x0)
    bq = f.sos.biquads
    n = length(bq)
    z0 = _zero_like(x0)
    T = typeof(z0)
    ButterState{T}(
        [b.b0 for b in bq], [b.b1 for b in bq], [b.b2 for b in bq],
        [b.a1 for b in bq], [b.a2 for b in bq],
        Float64(f.sos.g),
        T[_zero_like(x0) for _ in 1:n], T[_zero_like(x0) for _ in 1:n],
        _zero_like(x0),
    )
end

_zero_like(x::Number) = zero(float(x))
_zero_like(x::AbstractArray) = zeros(float(eltype(x)), size(x))

"""
    step!(state, x) -> y

Push one sample `x` through the streaming filter and return the filtered sample.
For array samples the returned array is owned by the state and is overwritten on
the next call — `copy` it if you need to keep it.
"""
function step!(s::ButterState{<:Number}, x::Number)
    v = float(x) * s.gain
    @inbounds for k in eachindex(s.b0)
        y = s.b0[k] * v + s.z1[k]
        s.z1[k] = s.b1[k] * v - s.a1[k] * y + s.z2[k]
        s.z2[k] = s.b2[k] * v - s.a2[k] * y
        v = y
    end
    return v
end

function step!(s::ButterState{<:AbstractArray}, x::AbstractArray)
    size(x) == size(s.buf) || throw(DimensionMismatch("sample size $(size(x)) != state size $(size(s.buf))"))
    v = s.buf
    g = s.gain
    @. v = x * g
    @inbounds for k in eachindex(s.b0)
        b0, b1, b2, a1, a2 = s.b0[k], s.b1[k], s.b2[k], s.a1[k], s.a2[k]
        z1, z2 = s.z1[k], s.z2[k]
        # y = b0*v + z1 ; z1 = b1*v - a1*y + z2 ; z2 = b2*v - a2*y  (element-wise, in place)
        for i in eachindex(v, z1, z2)
            xi = v[i]
            y = b0 * xi + z1[i]
            z1[i] = b1 * xi - a1 * y + z2[i]
            z2[i] = b2 * xi - a2 * y
            v[i] = y
        end
    end
    return v
end

# ---------------------------------------------------------------------------
# Difference of two first-order low-passes (EVM, Wu et al. 2012)
# ---------------------------------------------------------------------------

"""
    DiffLowpass(band::BandSpec, fs)

Temporal band-pass formed as `lowpass(f_hi) - lowpass(f_lo)` with two first-order IIR
low-passes. Cheap, per-pixel, streaming only — the filter used for real-time EVM.
"""
struct DiffLowpass
    band::BandSpec
    fs::Float64
    r_hi::Float64   # smoothing coefficient for the high cut-off low-pass
    r_lo::Float64   # smoothing coefficient for the low cut-off low-pass
end

_lp_coef(fc, fs) = 1 - exp(-2π * fc / fs)

function DiffLowpass(band::BandSpec, fs::Real)
    _check_band(band, fs)
    DiffLowpass(band, Float64(fs), _lp_coef(band.hi, fs), _lp_coef(band.lo, fs))
end

mutable struct DiffLowpassState{T}
    r_hi::Float64
    r_lo::Float64
    lp_hi::T
    lp_lo::T
    out::T
    initialized::Bool
end

function init_state(f::DiffLowpass, x0)
    DiffLowpassState(f.r_hi, f.r_lo, _zero_like(x0), _zero_like(x0), _zero_like(x0), false)
end

function step!(s::DiffLowpassState{<:Number}, x::Number)
    xf = float(x)
    if !s.initialized
        s.lp_hi = xf; s.lp_lo = xf; s.initialized = true   # start at the DC level: no step transient
    end
    s.lp_hi += s.r_hi * (xf - s.lp_hi)
    s.lp_lo += s.r_lo * (xf - s.lp_lo)
    s.out = s.lp_hi - s.lp_lo
    return s.out
end

function step!(s::DiffLowpassState{<:AbstractArray}, x::AbstractArray)
    size(x) == size(s.out) || throw(DimensionMismatch("sample size $(size(x)) != state size $(size(s.out))"))
    if !s.initialized
        s.lp_hi .= x; s.lp_lo .= x; s.initialized = true
    end
    rh, rl = s.r_hi, s.r_lo
    @. s.lp_hi += rh * (x - s.lp_hi)
    @. s.lp_lo += rl * (x - s.lp_lo)
    @. s.out = s.lp_hi - s.lp_lo
    return s.out
end

# ---------------------------------------------------------------------------
# Ideal FFT band-pass (batch)
# ---------------------------------------------------------------------------

"""
    ideal_bandpass(x, band::BandSpec, fs; dims=1)

Ideal (brick-wall) band-pass: FFT along `dims`, zero every bin outside
`band.lo <= |f| <= band.hi`, inverse FFT. Used for offline EVM.
"""
function ideal_bandpass(x::AbstractArray{<:Real}, band::BandSpec, fs::Real; dims::Integer=1)
    n = size(x, dims)
    X = rfft(float.(x), dims)
    freqs = rfftfreq(n, fs)
    keep = (band.lo .<= freqs) .& (freqs .<= band.hi)
    shp = ntuple(d -> d == dims ? length(freqs) : 1, ndims(x))
    X .*= reshape(keep, shp)
    return irfft(X, n, dims)
end

end # module TemporalFilters
