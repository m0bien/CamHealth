# Heart-rate estimation: pulse signal -> bpm over time with an SNR quality score.

using DSP: welch_pgram, freq, power, hanning
using Statistics: mean, median

"""
    welch_psd(x, fs; seglen=length(x), nfft=4096) -> (f, P)

Welch power spectrum (Hann window, 50% overlap) with zero padding to `nfft`.
"""
function welch_psd(x::AbstractVector{<:Real}, fs::Real; seglen::Integer=length(x), nfft::Integer=4096)
    seglen = min(seglen, length(x))
    p = welch_pgram(Float64.(x), seglen, seglen ÷ 2; fs=fs, nfft=max(nfft, seglen), window=hanning)
    return collect(freq(p)), power(p)
end

"""
    spectrum_peak(f, P, band) -> (fpeak, index)

Highest *local maximum* of `P` with `band.lo <= f <= band.hi`, refined by parabolic
interpolation through the peak bin and its neighbours. Only true peaks count: the
band-pass skirt often leaves the largest value on the band edge (e.g. 42 bpm), which
is not a pulse. Falls back to the in-band maximum when there is no local maximum.
"""
function spectrum_peak(f::AbstractVector, P::AbstractVector, band::BandSpec)
    idx = findall(x -> band.lo <= x <= band.hi, f)
    isempty(idx) && return NaN, 0
    peaks = [i for i in idx if 1 < i < length(P) && P[i] > P[i-1] && P[i] >= P[i+1]]
    cand = isempty(peaks) ? idx : peaks
    k = cand[argmax(@view P[cand])]
    fpk = f[k]
    if 1 < k < length(P)
        a, b, c = P[k-1], P[k], P[k+1]
        den = a - 2b + c
        if den < 0
            δ = 0.5 * (a - c) / den
            fpk += clamp(δ, -0.5, 0.5) * (f[2] - f[1])
        end
    end
    return fpk, k
end

"""
    snr_db(f, P, f0; band=PULSE, tol=0.1)

SNR after de Haan & Jeanne (2013): power within ±`tol` Hz (use at least the window main-lobe
half-width, `2fs/seglen` for Hann) of the pulse frequency `f0`
and of its first harmonic `2f0`, over the remaining power in the pulse band (extended
to cover the harmonic, capped at Nyquist).
"""
function snr_db(f::AbstractVector, P::AbstractVector, f0::Real; band::BandSpec=PULSE, tol::Real=0.1)
    isfinite(f0) || return NaN
    hi = min(max(band.hi, 2f0 + tol), f[end])
    inb = (f .>= band.lo) .& (f .<= hi)
    sig = inb .& ((abs.(f .- f0) .<= tol) .| (abs.(f .- 2f0) .<= tol))
    ps = sum(P[sig]); pn = sum(P[inb .& .!sig])
    (ps > 0 && pn > 0) || return NaN
    return 10log10(ps / pn)
end

"""Heart-rate track: window centre times (s), bpm, SNR (dB) and fraction of valid samples."""
struct HRTrack
    t::Vector{Float64}
    bpm::Vector{Float64}
    snr::Vector{Float64}
    valid_frac::Vector{Float64}
end

function Base.show(io::IO, h::HRTrack)
    ok = isfinite.(h.bpm)
    print(io, "HRTrack($(length(h.t)) windows, median $(round(median_or_nan(h.bpm[ok]), digits=1)) bpm, ",
          "median SNR $(round(median_or_nan(h.snr[ok]), digits=1)) dB)")
end
median_or_nan(x) = isempty(x) ? NaN : median(x)

"""Remove a least-squares straight line."""
function detrend_linear(x::AbstractVector{<:Real})
    n = length(x); τ = collect(1.0:n) .- (n + 1) / 2
    x .- mean(x) .- τ .* (sum(τ .* x) / sum(abs2, τ))
end

"""
    estimate_hr(signal, fs; window=10, hop=1, band=PULSE, valid=trues, min_valid=0.8,
                seglen_s=8, nfft=4096, order=3) -> HRTrack

Detrend and band-pass (zero-phase Butterworth over `band`) the whole pulse signal, then
for each `window`-second window every `hop` seconds: Welch spectrum, peak in `band`
with parabolic interpolation -> bpm, plus SNR. Windows with less than `min_valid` valid
samples give `NaN` bpm.
"""
function estimate_hr(x::AbstractVector{<:Real}, fs::Real; window::Real=10, hop::Real=1,
                     band::BandSpec=PULSE, valid::AbstractVector{Bool}=trues(length(x)),
                     min_valid::Real=0.8, seglen_s::Real=8, nfft::Integer=4096, order::Integer=3)
    y = apply(ButterBandpass(band, fs; order=order), detrend_linear(Float64.(x)))
    W = round(Int, window * fs); H = max(1, round(Int, hop * fs))
    seglen = min(W, round(Int, seglen_s * fs))
    tol = 2fs / seglen                      # Hann main-lobe half-width: counts the whole peak as signal
    ts = Float64[]; bpm = Float64[]; snr = Float64[]; vf = Float64[]
    length(y) < W && return HRTrack(ts, bpm, snr, vf)
    for s in 1:H:length(y)-W+1
        r = s:s+W-1
        frac = mean(@view valid[r])
        push!(ts, (s - 1 + W / 2) / fs); push!(vf, frac)
        if frac < min_valid
            push!(bpm, NaN); push!(snr, NaN); continue
        end
        f, P = welch_psd(y[r], fs; seglen=seglen, nfft=nfft)
        f0, _ = spectrum_peak(f, P, band)
        push!(bpm, 60 * f0); push!(snr, snr_db(f, P, f0; band=band, tol=tol))
    end
    return HRTrack(ts, bpm, snr, vf)
end
