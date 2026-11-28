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

"""
Heart-rate track: window centre times `t` (s), raw peak `bpm`, `snr` (dB), fraction of
valid (face present) samples, and `ok` — the quality gate: enough valid samples,
SNR ≥ threshold and no implausible jump from the previous accepted estimate. Use
`hr.bpm[hr.ok]` for reporting; rejected windows are "low quality".
"""
struct HRTrack
    t::Vector{Float64}
    bpm::Vector{Float64}
    snr::Vector{Float64}
    valid_frac::Vector{Float64}
    ok::BitVector
end
HRTrack(t, bpm, snr, vf) = HRTrack(t, bpm, snr, vf, BitVector(isfinite.(bpm)))

"""Median bpm over windows that passed the quality gate (`NaN` if none)."""
median_bpm(h::HRTrack) = median_or_nan(h.bpm[h.ok])
median_snr(h::HRTrack) = median_or_nan(h.snr[h.ok])

function Base.show(io::IO, h::HRTrack)
    print(io, "HRTrack($(length(h.t)) windows, $(count(h.ok)) good, median $(round(median_bpm(h), digits=1)) bpm, ",
          "median SNR $(round(median_snr(h), digits=1)) dB)")
end
median_or_nan(x) = isempty(x) ? NaN : median(x)

"""Remove a least-squares straight line."""
function detrend_linear(x::AbstractVector{<:Real})
    n = length(x); τ = collect(1.0:n) .- (n + 1) / 2
    x .- mean(x) .- τ .* (sum(τ .* x) / sum(abs2, τ))
end

"""Detrended, zero-phase band-passed pulse signal (what `estimate_hr` analyses)."""
filter_pulse(x::AbstractVector{<:Real}, fs::Real; band::BandSpec=PULSE, order::Integer=3) =
    apply(ButterBandpass(band, fs; order=order), detrend_linear(Float64.(x)))

"""
    estimate_hr(signal, fs; window=10, hop=1, band=PULSE, valid=trues, min_valid=0.8,
                seglen_s=window, nfft=4096, order=3, snr_tol=0.1, snr_min=-3, max_jump=15) -> HRTrack

Detrend and band-pass (zero-phase Butterworth over `band`) the whole pulse signal, then
for each `window`-second window every `hop` seconds: Welch spectrum, peak in `band`
with parabolic interpolation -> bpm, and SNR (de Haan: ±`snr_tol` Hz around the peak
and its first harmonic vs the rest of the band; 0.1 Hz = ±6 bpm).

Quality gate (`hr.ok`): a window is rejected when it has less than `min_valid` valid
samples (`bpm = NaN`), when its SNR is below `snr_min` dB, or when its bpm differs from
the previous accepted window by more than `max_jump` bpm per second of hop (after 3
such rejections in a row the new rate is accepted, so one bad early estimate cannot
lock the track).
"""
function estimate_hr(x::AbstractVector{<:Real}, fs::Real; window::Real=10, hop::Real=1,
                     band::BandSpec=PULSE, valid::AbstractVector{Bool}=trues(length(x)),
                     min_valid::Real=0.8, seglen_s::Real=window, nfft::Integer=4096, order::Integer=3,
                     snr_tol::Real=0.1, snr_min::Real=-3, max_jump::Real=15)
    y = filter_pulse(x, fs; band=band, order=order)
    W = round(Int, window * fs); H = max(1, round(Int, hop * fs))
    seglen = min(W, round(Int, seglen_s * fs))
    ts = Float64[]; bpm = Float64[]; snr = Float64[]; vf = Float64[]; ok = Bool[]
    length(y) < W && return HRTrack(ts, bpm, snr, vf, BitVector(ok))
    last_ok = NaN; last_t = NaN; njump = 0
    for s in 1:H:length(y)-W+1
        r = s:s+W-1
        frac = mean(@view valid[r])
        t = (s - 1 + W / 2) / fs
        push!(ts, t); push!(vf, frac)
        if frac < min_valid
            push!(bpm, NaN); push!(snr, NaN); push!(ok, false); continue
        end
        f, P = welch_psd(y[r], fs; seglen=seglen, nfft=nfft)
        f0, _ = spectrum_peak(f, P, band)
        b = 60 * f0; q = snr_db(f, P, f0; band=band, tol=snr_tol)
        good = isfinite(b) && isfinite(q) && q >= snr_min
        if good && isfinite(last_ok) && abs(b - last_ok) > max_jump * max(t - last_t, hop)
            njump += 1
            good = njump >= 3          # 3 consistent "jumps" in a row: re-lock on the new rate
        end
        good && (njump = 0)
        if good
            last_ok = b; last_t = t
        end
        push!(bpm, b); push!(snr, q); push!(ok, good)
    end
    return HRTrack(ts, bpm, snr, vf, BitVector(ok))
end
