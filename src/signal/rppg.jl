# rPPG methods: RGB trace (N x 3, uniformly sampled) -> 1-D pulse signal.
# M1: green channel. TODO (M2): CHROM (de Haan & Jeanne 2013), POS (Wang et al. 2017).

using Statistics: mean

"""
    rppg_green(rgb; valid=trues(size(rgb, 1))) -> Vector{Float64}

Green-channel method (Verkruysse et al. 2008): the green mean normalised by its average
over valid samples, minus 1 (relative change). Invalid samples are set to 0 after
normalisation so they do not inject steps into the filter.
"""
function rppg_green(rgb::AbstractMatrix{<:Real}; valid::AbstractVector{Bool}=trues(size(rgb, 1)))
    g = Float64.(rgb[:, 2])
    μ = mean(g[valid])
    s = g ./ μ .- 1
    s[.!valid] .= 0
    return s
end

"""Dispatch on a method name (`:green`; `:chrom`, `:pos` arrive in M2)."""
function rppg(rgb::AbstractMatrix{<:Real}, method::Symbol=:green; kw...)
    method === :green && return rppg_green(rgb; kw...)
    error("rPPG method $method not implemented yet (available: :green)")
end
