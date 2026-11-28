# rPPG methods: RGB trace (N x 3, uniformly sampled, columns R, G, B) -> 1-D pulse signal.
#
# Sign convention: every method returns a "PPG-like" signal whose maxima are the systolic
# peaks (most blood in the skin = least reflected light), so beats can be detected the same
# way for every method and compared with a contact PPG.
#
#   green  Verkruysse et al. 2008
#   chrom  de Haan & Jeanne 2013, "Robust pulse rate from chrominance-based rPPG"
#   pos    Wang, den Brinker, Stuijk & de Haan 2017, "Algorithmic principles of rPPG" (default)

using Statistics: mean, std
using DSP: hanning

const RPPG_METHODS = (:green, :chrom, :pos)

"""
    rppg_green(rgb; valid=trues(size(rgb, 1))) -> Vector{Float64}

Green-channel method: `-(G / mean(G) - 1)` over valid samples (inverted so systole is a
maximum). Invalid samples are 0.
"""
function rppg_green(rgb::AbstractMatrix{<:Real}; valid::AbstractVector{Bool}=trues(size(rgb, 1)))
    g = Float64.(rgb[:, 2])
    μ = mean(g[valid])
    s = 1 .- g ./ μ
    s[.!valid] .= 0
    return s
end

# windows of length L (samples) with hop L÷2 that contain only valid samples
function _windows(n::Integer, L::Integer, valid::AbstractVector{Bool})
    starts = 1:max(1, L ÷ 2):max(1, n - L + 1)
    [s:s+L-1 for s in starts if s + L - 1 <= n && all(@view valid[s:s+L-1])]
end

"""
    rppg_chrom(rgb, fs; valid, window=1.6, band=PULSE, order=3) -> Vector{Float64}

CHROM (de Haan & Jeanne 2013). Per window of `window` s (50 % overlap): normalise RGB by
the window mean, `Xs = 3R - 2G`, `Ys = 1.5R + G - 1.5B`, band-pass both, then
`S = Xf - α Yf` with `α = std(Xf)/std(Yf)`; Hann-weighted overlap-add. Cancels intensity
changes and (white) specular reflections.
"""
function rppg_chrom(rgb::AbstractMatrix{<:Real}, fs::Real; valid::AbstractVector{Bool}=trues(size(rgb, 1)),
                    window::Real=1.6, band::BandSpec=PULSE, order::Integer=3)
    n = size(rgb, 1)
    L = max(4, 2 * round(Int, window * fs / 2))          # even window length
    f = ButterBandpass(band, fs; order=order)
    hw = hanning(L)
    out = zeros(n)
    for r in _windows(n, L, valid)
        C = Float64.(rgb[r, :])
        Cn = C ./ mean(C, dims=1)
        Xs = 3 .* Cn[:, 1] .- 2 .* Cn[:, 2]
        Ys = 1.5 .* Cn[:, 1] .+ Cn[:, 2] .- 1.5 .* Cn[:, 3]
        Xf = apply(f, Xs); Yf = apply(f, Ys)
        sy = std(Yf)
        α = sy > 0 ? std(Xf) / sy : 0.0
        S = Xf .- α .* Yf
        out[r] .+= hw .* (S .- mean(S))
    end
    return out
end

"""
    rppg_pos(rgb, fs; valid, window=1.6) -> Vector{Float64}

Plane-Orthogonal-to-Skin (Wang et al. 2017). Per window of `window` s (one-sample hop, as
in the paper): `Cn = C / mean(C)`, `S1 = G - B`, `S2 = -2R + G + B`,
`h = S1 + (std(S1)/std(S2)) S2`, overlap-added after removing its mean. Returned negated so
systole is a maximum.
"""
function rppg_pos(rgb::AbstractMatrix{<:Real}, fs::Real; valid::AbstractVector{Bool}=trues(size(rgb, 1)),
                  window::Real=1.6)
    n = size(rgb, 1)
    L = max(2, ceil(Int, window * fs))
    C = Float64.(rgb)
    H = zeros(n)
    @inbounds for s in 1:n-L+1
        e = s + L - 1
        all(@view valid[s:e]) || continue
        mR = mean(@view C[s:e, 1]); mG = mean(@view C[s:e, 2]); mB = mean(@view C[s:e, 3])
        S1 = [C[k, 2] / mG - C[k, 3] / mB for k in s:e]
        S2 = [-2C[k, 1] / mR + C[k, 2] / mG + C[k, 3] / mB for k in s:e]
        σ2 = std(S2)
        h = σ2 > 0 ? S1 .+ (std(S1) / σ2) .* S2 : S1
        h .-= mean(h)
        H[s:e] .+= h
    end
    return -H
end

"""
    rppg(rgb, method=:pos; fs=30, valid=trues) -> Vector{Float64}

Dispatch on the method name: `:green`, `:chrom` or `:pos`.
"""
function rppg(rgb::AbstractMatrix{<:Real}, method::Symbol=:pos; fs::Real=30,
              valid::AbstractVector{Bool}=trues(size(rgb, 1)))
    method === :green && return rppg_green(rgb; valid=valid)
    method === :chrom && return rppg_chrom(rgb, fs; valid=valid)
    method === :pos && return rppg_pos(rgb, fs; valid=valid)
    error("unknown rPPG method $method (available: $(join(RPPG_METHODS, ", ")))")
end
