# Benchmark: rPPG methods vs contact PPG, window by window and beat by beat.

using Statistics: mean, median, std, cor
using DelimitedFiles: readdlm, writedlm
using Printf: @sprintf

"""
    hr_metrics(est, ref) -> NamedTuple (n, mae, rmse, r, within5, ba_bias, ba_lo, ba_hi)

Agreement between paired bpm estimates (`NaN` pairs dropped). Bland–Altman bias and 95 %
limits of agreement are on `est - ref`.
"""
function hr_metrics(est::AbstractVector{<:Real}, ref::AbstractVector{<:Real})
    k = isfinite.(est) .& isfinite.(ref)
    e = Float64.(est[k]); r = Float64.(ref[k]); d = e .- r
    n = length(d)
    n == 0 && return (n=0, mae=NaN, rmse=NaN, r=NaN, within5=NaN, ba_bias=NaN, ba_lo=NaN, ba_hi=NaN)
    sd = n > 1 ? std(d) : 0.0
    rr = n > 2 && std(e) > 0 && std(r) > 0 ? cor(e, r) : NaN
    return (n=n, mae=mean(abs.(d)), rmse=sqrt(mean(abs2, d)), r=rr, within5=mean(abs.(d) .<= 5),
            ba_bias=mean(d), ba_lo=mean(d) - 1.96sd, ba_hi=mean(d) + 1.96sd)
end

"""
    beat_metrics(est::Beats, ref::Beats; tol=0.15) -> NamedTuple (matched, ibi_err_ms, rmssd_est, rmssd_ref, rmssd_err_ms)

Match estimated peaks to reference peaks within `tol` s; IBI error = mean |ΔIBI| over
consecutive matched pairs whose estimated IBI passed cleaning.
"""
function beat_metrics(est::Beats, ref::Beats; tol::Real=0.15)
    m = Int[]                                      # m[k] = matched ref index for est peak k (0 = none)
    for p in est.peaks
        isempty(ref.peaks) && (push!(m, 0); continue)
        j = argmin(abs.(ref.peaks .- p))
        push!(m, abs(ref.peaks[j] - p) <= tol ? j : 0)
    end
    errs = Float64[]
    for k in 1:length(est.peaks)-1
        if m[k] > 0 && m[k+1] == m[k] + 1 && est.ibi_ok[k]
            push!(errs, 1000 * abs(est.ibi[k] - ref.ibi[m[k]]))
        end
    end
    he = hrv_metrics(est); hr_ = hrv_metrics(ref)
    return (matched=count(>(0), m) / max(1, length(ref.peaks)), ibi_err_ms=isempty(errs) ? NaN : mean(errs),
            rmssd_est=he.rmssd_ms, rmssd_ref=hr_.rmssd_ms, rmssd_err_ms=abs(he.rmssd_ms - hr_.rmssd_ms))
end

"""Cached ROI trace for a sample (`results/cache/<tag>_<name>.csv`; tag = benchmark run name)."""
function sample_trace(s::BenchSample, cachedir::AbstractString; roi=:yunet, verbose=true, tag=s.dataset)
    f = joinpath(cachedir, "$(tag)_$(s.name).csv")
    if isfile(f)
        d = readdlm(f, ',', Float64; skipstart=1)
        return RGBTrace(d[:, 1], d[:, 2:4], BitVector(d[:, 5] .> 0), round.(Int, d[:, 6]))
    end
    tracker = roi === :yunet ? FaceTracker(YuNetDetector(); every=10) : roi
    src = s.open_frames()
    tr = try extract_trace(src, tracker; verbose=false) finally close(src) end
    mkpath(cachedir)
    open(f, "w") do io
        println(io, "t,r,g,b,valid,npix")
        writedlm(io, hcat(tr.t, tr.rgb, Int.(tr.valid), tr.npix), ',')
    end
    verbose && println("  $(s.name): $(length(tr)) frames, face in $(count(tr.valid))")
    return tr
end

"""Reference HR track and beats from the contact PPG, on the same time grid as `res`."""
function reference(s::BenchSample, res::PulseResult; window=10, hop=1)
    fs = res.fs
    k = [true; diff(s.ppg_t) .> 0]
    itp = linear_interpolation(s.ppg_t[k], s.ppg[k]; extrapolation_bc=Flat())
    p = itp.(res.t)
    inside = (res.t .>= s.ppg_t[1]) .& (res.t .<= s.ppg_t[end])
    hr = estimate_hr(p, fs; window=window, hop=hop, valid=inside, snr_min=-Inf, max_jump=Inf)
    beats = detect_beats(filter_pulse(p, fs), fs; valid=inside, hr=hr)
    return hr, beats
end

"""
    run_benchmark(samples; methods=(:green, :chrom, :pos), roi=:yunet, fs=30, snr_min=-3,
                  outdir="results", tag="dataset") -> (rows, summary)

For every sample: ROI trace (cached), every rPPG method, HR track and beats; compare with the
reference PPG window by window (same 10 s windows) and beat by beat. Writes
`<outdir>/benchmark_<tag>.csv` (one row per sample × method) and appends a summary section
to `<outdir>/README.md`.
"""
function run_benchmark(samples::Vector{BenchSample}; methods=RPPG_METHODS, roi=:yunet, fs::Real=30,
                       snr_min::Real=-3, outdir::AbstractString="results", tag::AbstractString="dataset",
                       verbose::Bool=true)
    mkpath(outdir)
    rows = NamedTuple[]
    pooled = Dict(m => (est=Float64[], ref=Float64[], est_all=Float64[], ref_all=Float64[]) for m in methods)
    for s in samples
        tr = sample_trace(s, joinpath(outdir, "cache"); roi=roi, verbose=verbose, tag=tag)
        count(tr.valid) < 2 && (verbose && println("  $(s.name): no face, skipped"); continue)
        for m in methods
            res = analyze_trace(tr; method=m, fs=fs, snr_min=snr_min, hrv=true)
            refhr, refbeats = reference(s, res)
            n = min(length(res.hr.t), length(refhr.t))
            est = res.hr.bpm[1:n]; ref = refhr.bpm[1:n]; ok = res.hr.ok[1:n]
            g = hr_metrics(est[ok], ref[ok]); a = hr_metrics(est, ref)
            b = beat_metrics(res.beats, refbeats)
            append!(pooled[m].est, est[ok]); append!(pooled[m].ref, ref[ok])
            append!(pooled[m].est_all, est); append!(pooled[m].ref_all, ref)
            push!(rows, (sample=s.name, method=m, windows=n, coverage=n > 0 ? count(ok) / n : NaN,
                         mae=g.mae, rmse=g.rmse, r=g.r, within5=g.within5, ba_bias=g.ba_bias, ba_lo=g.ba_lo, ba_hi=g.ba_hi,
                         mae_all=a.mae, ref_bpm=median_or_nan(filter(isfinite, ref)), est_bpm=median_or_nan(est[ok]),
                         beats_matched=b.matched, ibi_err_ms=b.ibi_err_ms, rmssd_ref=b.rmssd_ref,
                         rmssd_est=b.rmssd_est, rmssd_err_ms=b.rmssd_err_ms))
            verbose && println(@sprintf("  %-12s %-5s ref %5.1f est %5.1f  MAE %5.2f  coverage %3.0f%%  IBI err %5.1f ms",
                                        s.name, m, rows[end].ref_bpm, rows[end].est_bpm, g.mae, 100rows[end].coverage, b.ibi_err_ms))
        end
    end
    csv = joinpath(outdir, "benchmark_$(tag).csv")
    if !isempty(rows)
        ks = keys(rows[1])
        open(csv, "w") do io
            println(io, join(ks, ","))
            for r in rows
                println(io, join([v isa AbstractFloat ? (isfinite(v) ? string(round(v, digits=4)) : "") : string(v) for v in values(r)], ","))
            end
        end
    end
    summary = [(method=m, pooled=hr_metrics(pooled[m].est, pooled[m].ref), pooled_all=hr_metrics(pooled[m].est_all, pooled[m].ref_all),
                coverage=length(pooled[m].est) / max(1, length(pooled[m].est_all)),
                subj_mae=mean_finite([r.mae for r in rows if r.method == m]),
                ibi_err=mean_finite([r.ibi_err_ms for r in rows if r.method == m]),
                rmssd_err=mean_finite([r.rmssd_err_ms for r in rows if r.method == m])) for m in methods]
    write_summary_md(joinpath(outdir, "README.md"), tag, length(unique(r.sample for r in rows)), summary, snr_min)
    verbose && println("wrote $csv and $(joinpath(outdir, "README.md"))")
    return rows, summary
end

mean_finite(x) = (y = filter(isfinite, x); isempty(y) ? NaN : mean(y))

function write_summary_md(path, tag, nsamples, summary, snr_min)
    f(x, d=2) = isfinite(x) ? string(round(x, digits=d)) : "–"
    section = IOBuffer()
    println(section, "## $(tag)\n")
    println(section, "$(nsamples) recordings, 10 s windows / 1 s hop, quality gate SNR ≥ $(snr_min) dB. ",
            "Errors are pooled over all gated windows; *all windows* ignores the gate.\n")
    println(section, "| method | MAE (bpm) | RMSE | Pearson r | within 5 bpm | Bland–Altman bias [95 % LoA] | coverage | MAE all windows | per-recording MAE | IBI error (ms) | RMSSD error (ms) |")
    println(section, "|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summary
        p = s.pooled
        println(section, "| $(s.method) | $(f(p.mae)) | $(f(p.rmse)) | $(f(p.r, 3)) | $(f(100p.within5, 1)) % | $(f(p.ba_bias)) [$(f(p.ba_lo)), $(f(p.ba_hi))] | ",
                "$(f(100s.coverage, 1)) % | $(f(s.pooled_all.mae)) | $(f(s.subj_mae)) | $(f(s.ibi_err, 1)) | $(f(s.rmssd_err, 1)) |")
    end
    println(section)
    text = String(take!(section))
    # replace this tag's section if present, keep the others
    old = isfile(path) ? read(path, String) : "# Benchmark results\n\nGenerated by `scripts/benchmark.jl`.\n\n"
    rx = Regex("## \\Q$(tag)\\E\\n.*?(?=\\n## |\\z)", "s")
    new = occursin(rx, old) ? replace(old, rx => rstrip(text)) : rstrip(old) * "\n\n" * text
    write(path, rstrip(new) * "\n")
end
