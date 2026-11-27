# Offline pipeline and command-line entry point.
#
#   julia --project scripts/campulse.jl clip.mkv                         # POS, quality-gated bpm
#   julia --project scripts/campulse.jl clip.mkv --method chrom --csv out.csv --hrv
#   julia --project scripts/campulse.jl clip.mkv --roi 260,120,120,80      # manual ROI, no detector

using Statistics: median, mean

"""Result of the signal stage for one rPPG method."""
struct PulseResult
    method::Symbol
    fs::Float64
    t::Vector{Float64}          # uniform time grid (s)
    signal::Vector{Float64}     # raw rPPG signal
    pulse::Vector{Float64}      # detrended + band-passed (PPG-like, systole = max)
    valid::BitVector
    hr::HRTrack
    beats::Union{Nothing,Beats}
end

"""
    analyze_trace(trace::RGBTrace; method=:pos, fs=30, window=10, hop=1, snr_min=-3, hrv=false) -> PulseResult

Resample a colour trace, apply the rPPG `method`, estimate the heart-rate track (with the
SNR / jump quality gate) and, with `hrv=true`, detect beats.
"""
function analyze_trace(trace::RGBTrace; method::Symbol=:pos, fs::Real=30, window::Real=10, hop::Real=1,
                       snr_min::Real=-3, hrv::Bool=false)
    count(trace.valid) >= 2 || error("no face/ROI samples in trace")
    tu, X, vu = resample_uniform(trace, fs)
    sig = rppg(X, method; fs=fs, valid=vu)
    hr = estimate_hr(sig, fs; window=window, hop=hop, valid=vu, snr_min=snr_min)
    y = filter_pulse(sig, fs)
    beats = hrv ? detect_beats(y, fs; valid=vu, hr=hr) : nothing
    return PulseResult(method, fs, tu, sig, y, BitVector(vu), hr, beats)
end

"""
    clip_trace(path; roi=:yunet, fs=30, verbose=true) -> (trace, monitor)

Frames of a video file -> ROI colour trace (YuNet face tracker or a `FixedROI`).
"""
function clip_trace(path::AbstractString; roi=:yunet, fs::Real=30, verbose::Bool=true,
                    maxframes::Integer=typemax(Int))
    tracker = roi === :yunet ? FaceTracker(YuNetDetector(); every=10) : roi
    mon = FrameMonitor(nominal_fps=fs, warn=false)
    src = FileSource(path; monitor=mon)
    trace = try
        extract_trace(src, tracker; maxframes=maxframes, verbose=verbose)
    finally
        close(src)
    end
    verbose && println("frames: $(length(trace)), with face/ROI: $(count(trace.valid)), ", mon)
    count(trace.valid) >= 2 || error("no face/ROI found in $path")
    return trace, mon
end

"""Write the per-window HR track: `time_s,bpm,snr_db,valid_frac,quality` (bpm empty when low quality)."""
function write_hr_csv(path::AbstractString, hr::HRTrack)
    open(path, "w") do io
        println(io, "time_s,bpm,snr_db,valid_frac,quality")
        for k in eachindex(hr.t)
            q = hr.ok[k] ? "ok" : (hr.valid_frac[k] < 0.8 ? "no face" : "low quality")
            b = hr.ok[k] ? string(round(hr.bpm[k], digits=2)) : ""
            s = isfinite(hr.snr[k]) ? string(round(hr.snr[k], digits=2)) : ""
            println(io, round(hr.t[k], digits=2), ",", b, ",", s, ",", round(hr.valid_frac[k], digits=3), ",", q)
        end
    end
end

"""Write beats: `beat,peak_s,foot_s,ibi_ms,ibi_ok` (IBI ending at that beat)."""
function write_beats_csv(path::AbstractString, b::Beats)
    open(path, "w") do io
        println(io, "beat,peak_s,foot_s,ibi_ms,ibi_ok")
        for k in eachindex(b.peaks)
            ibi = k > 1 ? string(round(1000b.ibi[k-1], digits=1)) : ""
            ok = k > 1 ? string(Int(b.ibi_ok[k-1])) : ""
            ft = isfinite(b.feet[k]) ? string(round(b.feet[k], digits=4)) : ""
            println(io, k, ",", round(b.peaks[k], digits=4), ",", ft, ",", ibi, ",", ok)
        end
    end
end

"""
    analyze_clip(path; method=:pos, roi=:yunet, fs=30, window=10, hop=1, csv=nothing,
                 hrv=false, beats_csv=nothing, verbose=true) -> (hr, trace, monitor)

Full offline pipeline on a video file. See `analyze_trace` for the signal stage. Writes the
HR track to `csv` and, with `hrv`, beats to `beats_csv`.
"""
function analyze_clip(path::AbstractString; method::Symbol=:pos, roi=:yunet, fs::Real=30,
                      window::Real=10, hop::Real=1, csv=nothing, hrv::Bool=false, beats_csv=nothing,
                      snr_min::Real=-3, verbose::Bool=true, maxframes::Integer=typemax(Int))
    trace, mon = clip_trace(path; roi=roi, fs=fs, verbose=verbose, maxframes=maxframes)
    res = analyze_trace(trace; method=method, fs=fs, window=window, hop=hop, snr_min=snr_min, hrv=hrv)
    if csv !== nothing
        write_hr_csv(csv, res.hr); verbose && println("wrote $csv")
    end
    if hrv && beats_csv !== nothing
        write_beats_csv(beats_csv, res.beats); verbose && println("wrote $beats_csv")
    end
    return res.hr, trace, mon, res
end

function _parse_roi(s::AbstractString)
    v = parse.(Float64, split(s, ','))
    length(v) == 4 || error("--roi expects x,y,w,h")
    FixedROI(v...)
end

const CLI_USAGE = """
usage: julia --project scripts/campulse.jl CLIP [--method pos|chrom|green|all] [--csv OUT.csv]
                    [--hrv] [--roi x,y,w,h] [--snr-min -3] [--window 10] [--hop 1] [--fs 30]
  CLIP       video file (e.g. data/clip.mkv from scripts/record_clip.jl)
  --method   rPPG method: pos (default), chrom, green, or all (compare the three)
  --csv      per-window time_s,bpm,snr_db,valid_frac,quality (bpm empty when low quality)
  --hrv      detect beats, print HRV, write data/<clip>_beats.csv
  --snr-min  quality gate: windows with SNR below this (dB) are "low quality"
  --roi      fixed ROI rectangle in pixels instead of the YuNet face tracker
"""

function _summary(io, res::PulseResult)
    hr = res.hr
    nq = count(hr.ok); nface = count(hr.valid_frac .>= 0.8)
    println(io, "[$(res.method)]")
    println(io, "  windows     : $(length(hr.t)) ($nface with face, $nq passed quality gate, $(nface - nq) low quality)")
    if nq > 0
        b = hr.bpm[hr.ok]
        println(io, "  heart rate  : median $(round(median(b), digits=1)) bpm  (IQR $(round(quantile_(b, 0.25), digits=1))–$(round(quantile_(b, 0.75), digits=1)))")
        println(io, "  SNR         : median $(round(median(hr.snr[hr.ok]), digits=1)) dB over good windows; all windows $(round(median_or_nan(filter(isfinite, hr.snr)), digits=1)) dB")
    else
        println(io, "  heart rate  : low quality — no window passed the gate (median SNR $(round(median_or_nan(filter(isfinite, hr.snr)), digits=1)) dB)")
    end
    if res.beats !== nothing
        h = hrv_metrics(res.beats)
        println(io, "  beats       : $(h.n_beats), clean IBIs $(h.n_ibi)/$(length(res.beats.ibi))")
        println(io, "  HRV         : mean HR $(round(h.mean_hr, digits=1)) bpm, SDNN $(round(h.sdnn_ms, digits=1)) ms, RMSSD $(round(h.rmssd_ms, digits=1)) ms, pNN50 $(round(100h.pnn50, digits=1)) %")
    end
end

"""Command-line entry point (`scripts/campulse.jl`)."""
function cli_main(args::Vector{String}=ARGS)
    isempty(args) && (print(CLI_USAGE); return 1)
    path = nothing; methods = [:pos]; csv = nothing; roi = :yunet; hrv = false
    window = 10.0; hop = 1.0; fs = 30.0; snr_min = -3.0
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help"); print(CLI_USAGE); return 0
        elseif a == "--method"
            m = args[i+=1]; methods = m == "all" ? collect(RPPG_METHODS) : [Symbol(m)]
        elseif a == "--csv"; csv = args[i+=1]
        elseif a == "--hrv"; hrv = true
        elseif a == "--roi"; roi = _parse_roi(args[i+=1])
        elseif a == "--snr-min"; snr_min = parse(Float64, args[i+=1])
        elseif a == "--window"; window = parse(Float64, args[i+=1])
        elseif a == "--hop"; hop = parse(Float64, args[i+=1])
        elseif a == "--fs"; fs = parse(Float64, args[i+=1])
        elseif startswith(a, "-"); error("unknown option $a\n" * CLI_USAGE)
        else path = a
        end
        i += 1
    end
    path === nothing && error("no clip given\n" * CLI_USAGE)
    all(in(RPPG_METHODS), methods) || error("unknown method in $methods\n" * CLI_USAGE)
    trace, mon = clip_trace(path; roi=roi, fs=fs)
    stem = replace(path, r"\.[^.\\/]+$" => "")
    println("\n== $(basename(path)) ==")
    println("frames        : $(length(trace)), dropped ≈ $(mon.ndropped) in $(mon.ndrop_events) gaps")
    for m in methods
        res = analyze_trace(trace; method=m, fs=fs, window=window, hop=hop, snr_min=snr_min, hrv=hrv)
        _summary(stdout, res)
        sfx = length(methods) > 1 ? "_$(m)" : ""
        if csv !== nothing
            out = length(methods) > 1 ? replace(csv, r"\.csv$" => "") * sfx * ".csv" : csv
            write_hr_csv(out, res.hr); println("  wrote       : $out")
        end
        if hrv
            out = stem * "_beats" * sfx * ".csv"
            write_beats_csv(out, res.beats); println("  wrote       : $out")
        end
    end
    return 0
end

quantile_(x, p) = (s = sort(x); s[clamp(round(Int, 1 + p * (length(s) - 1)), 1, length(s))])
