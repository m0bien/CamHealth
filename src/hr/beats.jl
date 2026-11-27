# Beat detection and heart-rate variability (HRV).
#
#   y  = filter_pulse(signal, fs)                 # band-passed, PPG-like (systole = maximum)
#   b  = detect_beats(y, fs; valid, hr)           # systolic peaks + pulse feet, cleaned IBIs
#   hrv_metrics(b)                                # mean HR, SDNN, RMSSD, pNN50

using Interpolations: cubic_spline_interpolation
using Statistics: mean, median, std

"""
Detected beats. `peaks`: systolic peak times (s); `feet`: pulse-foot times (s, point of
steepest upstroke before each peak, `NaN` if not found); `ibi`: inter-beat intervals (s)
between consecutive peaks, timed at the later beat (`ibi_t`); `ibi_ok`: IBIs that passed
cleaning (valid signal, not in a low-quality span, within 20 % of the local median).
"""
struct Beats
    peaks::Vector{Float64}
    feet::Vector{Float64}
    ibi::Vector{Float64}
    ibi_t::Vector{Float64}
    ibi_ok::BitVector
end

Base.show(io::IO, b::Beats) =
    print(io, "Beats($(length(b.peaks)) beats, $(count(b.ibi_ok))/$(length(b.ibi)) IBIs kept)")

# parabolic vertex offset (in samples) through (−1,a), (0,b), (1,c)
function _parabolic(a, b, c)
    den = a - 2b + c
    den == 0 && return 0.0
    clamp(0.5 * (a - c) / den, -0.5, 0.5)
end

"""Running RR interval (s) at time τ from an HR track (good windows only), else `rr0`."""
function _rr_at(hr, τ, rr0)
    hr === nothing && return rr0
    ok = findall(hr.ok)
    isempty(ok) && return rr0
    k = ok[argmin(abs.(hr.t[ok] .- τ))]
    return 60 / hr.bpm[k]
end

"""
    detect_beats(y, fs; valid=trues, hr=nothing, up=250, min_spacing=0.6, max_dev=0.2,
                 local_n=5) -> Beats

Find systolic peaks in a band-passed, PPG-like pulse signal `y` sampled at `fs`:

1. upsample to `up` Hz with a cubic spline;
2. take local maxima above zero, strongest first, keeping one only if no stronger beat lies
   within `min_spacing` × the running RR interval (from `hr`, an `HRTrack`, or the dominant
   frequency of `y`);
3. refine each peak by a parabola through the 3 highest samples;
4. pulse foot = maximum slope of the upstroke between the previous beat and the peak;
5. clean the IBIs: drop those touching invalid samples, falling in a rejected HR window,
   or deviating more than `max_dev` from the median of the `local_n` IBIs on each side.
"""
function detect_beats(y::AbstractVector{<:Real}, fs::Real; valid::AbstractVector{Bool}=trues(length(y)),
                      hr=nothing, up::Real=250, min_spacing::Real=0.6, max_dev::Real=0.2, local_n::Integer=5)
    n = length(y)
    n >= 4 || return Beats(Float64[], Float64[], Float64[], Float64[], falses(0))
    t = range(0, step=1 / fs, length=n)
    itp = cubic_spline_interpolation(t, Float64.(y))
    tu = range(0, t[end], step=1 / up)
    yu = itp.(tu)
    vu = [valid[clamp(round(Int, τ * fs) + 1, 1, n)] for τ in tu]

    # fallback RR from the dominant frequency
    rr0 = let (f, P) = welch_psd(y, fs; seglen=min(n, round(Int, 10fs)))
        f0, _ = spectrum_peak(f, P, PULSE)
        isfinite(f0) ? 1 / f0 : 1.0
    end

    cand = [i for i in 2:length(yu)-1 if yu[i] > 0 && yu[i] >= yu[i-1] && yu[i] > yu[i+1] && vu[i]]
    sort!(cand; by=i -> -yu[i])
    acc = Int[]                                   # accepted, kept sorted
    for i in cand
        sp = min_spacing * _rr_at(hr, tu[i], rr0) * up
        j = searchsortedfirst(acc, i)
        near = (j <= length(acc) && acc[j] - i < sp) || (j > 1 && i - acc[j-1] < sp)
        near || insert!(acc, j, i)
    end

    dy = diff(yu) .* up                           # slope at tu[k] + 0.5/up
    peaks = Float64[]; feet = Float64[]
    for (m, i) in enumerate(acc)
        push!(peaks, tu[i] + _parabolic(yu[i-1], yu[i], yu[i+1]) / up)
        lo = m > 1 ? acc[m-1] : max(1, i - round(Int, 0.6 * _rr_at(hr, tu[i], rr0) * up))
        r = lo:i-1
        if length(r) >= 3
            k = r[argmax(@view dy[r])]
            off = 1 < k < length(dy) ? _parabolic(dy[k-1], dy[k], dy[k+1]) : 0.0
            push!(feet, tu[k] + (0.5 + off) / up)
        else
            push!(feet, NaN)
        end
    end

    ibi = diff(peaks); ibi_t = peaks[2:end]
    ok = trues(length(ibi))
    for k in eachindex(ibi)
        a = clamp(round(Int, peaks[k] * fs) + 1, 1, n); b = clamp(round(Int, peaks[k+1] * fs) + 1, 1, n)
        all(@view valid[a:b]) || (ok[k] = false)
        if hr !== nothing && !isempty(hr.t)
            w = argmin(abs.(hr.t .- (peaks[k] + peaks[k+1]) / 2))
            hr.ok[w] || (ok[k] = false)
        end
    end
    for k in eachindex(ibi)
        nb = max(1, k - local_n):min(length(ibi), k + local_n)
        nb_ok = [j for j in nb if ok[j] || j == k]
        med = median(ibi[nb_ok])
        abs(ibi[k] - med) > max_dev * med && (ok[k] = false)
    end
    return Beats(peaks, feet, ibi, ibi_t, ok)
end

"""
    hrv_metrics(b::Beats) -> NamedTuple (n_beats, n_ibi, mean_hr, mean_ibi_ms, sdnn_ms, rmssd_ms, pnn50)

Time-domain HRV over the cleaned IBIs. RMSSD and pNN50 use only successive differences
where both neighbouring IBIs were kept.
"""
function hrv_metrics(b::Beats)
    x = b.ibi[b.ibi_ok] .* 1000
    d = [1000 * (b.ibi[k+1] - b.ibi[k]) for k in 1:length(b.ibi)-1 if b.ibi_ok[k] && b.ibi_ok[k+1]]
    return (n_beats=length(b.peaks), n_ibi=length(x),
            mean_hr=isempty(x) ? NaN : 60_000 / mean(x),
            mean_ibi_ms=isempty(x) ? NaN : mean(x),
            sdnn_ms=length(x) > 1 ? std(x) : NaN,
            rmssd_ms=isempty(d) ? NaN : sqrt(mean(abs2, d)),
            pnn50=isempty(d) ? NaN : count(abs.(d) .> 50) / length(d))
end

hrv_metrics(ibi_ms::AbstractVector{<:Real}) =
    hrv_metrics(Beats(Float64[], Float64[], Float64.(ibi_ms) ./ 1000, Float64[], trues(length(ibi_ms))))
