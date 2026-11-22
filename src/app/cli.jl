# Offline pipeline and command-line entry point.
#
#   julia --project scripts/campulse.jl clip.mkv --method green --csv out.csv
#   julia --project scripts/campulse.jl clip.mkv --roi 260,120,120,80      # manual ROI, no detector

using DelimitedFiles: writedlm
using Statistics: median, mean

"""
    analyze_clip(path; method=:green, roi=:yunet, fs=30, window=10, hop=1, csv=nothing, verbose=true)

Run the offline pipeline on a video file: frames -> ROI (YuNet face tracker, or a
`FixedROI`) -> RGB trace -> uniform resampling at `fs` -> rPPG `method` -> windowed
heart rate. Returns `(hr::HRTrack, trace::RGBTrace, monitor::FrameMonitor)`. With `csv`,
writes `time_s,bpm,snr_db,valid_frac` per window.
"""
function analyze_clip(path::AbstractString; method::Symbol=:green, roi=:yunet, fs::Real=30,
                      window::Real=10, hop::Real=1, csv=nothing, verbose::Bool=true,
                      maxframes::Integer=typemax(Int))
    tracker = roi === :yunet ? FaceTracker(YuNetDetector(); every=10) : roi
    mon = FrameMonitor(nominal_fps=fs, warn=false)
    src = FileSource(path; monitor=mon)
    trace = try
        extract_trace(src, tracker; maxframes=maxframes, verbose=verbose)
    finally
        close(src)
    end
    nvalid = count(trace.valid)
    verbose && println("frames: $(length(trace)), with face/ROI: $nvalid, ", mon)
    nvalid >= 2 || error("no face/ROI found in $path")
    tu, X, vu = resample_uniform(trace, fs)
    sig = rppg(X, method; valid=vu)
    hr = estimate_hr(sig, fs; window=window, hop=hop, valid=vu)
    if csv !== nothing
        open(csv, "w") do io
            println(io, "time_s,bpm,snr_db,valid_frac")
            writedlm(io, hcat(round.(hr.t, digits=2), round.(hr.bpm, digits=2), round.(hr.snr, digits=2), round.(hr.valid_frac, digits=3)), ',')
        end
        verbose && println("wrote $csv")
    end
    return hr, trace, mon
end

function _parse_roi(s::AbstractString)
    v = parse.(Float64, split(s, ','))
    length(v) == 4 || error("--roi expects x,y,w,h")
    FixedROI(v...)
end

const CLI_USAGE = """
usage: julia --project scripts/campulse.jl CLIP [--method green] [--csv OUT.csv]
                                                [--roi x,y,w,h] [--window 10] [--hop 1] [--fs 30]
  CLIP      video file (e.g. data/clip.mkv from scripts/record_clip.jl)
  --method  rPPG method: green (CHROM/POS come in M2)
  --csv     write per-window time_s,bpm,snr_db,valid_frac
  --roi     fixed ROI rectangle in pixels instead of the YuNet face tracker
"""

"""Command-line entry point (`scripts/campulse.jl`)."""
function cli_main(args::Vector{String}=ARGS)
    isempty(args) && (print(CLI_USAGE); return 1)
    path = nothing; method = :green; csv = nothing; roi = :yunet
    window = 10.0; hop = 1.0; fs = 30.0
    i = 1
    while i <= length(args)
        a = args[i]
        if a in ("-h", "--help"); print(CLI_USAGE); return 0
        elseif a == "--method"; method = Symbol(args[i+=1])
        elseif a == "--csv"; csv = args[i+=1]
        elseif a == "--roi"; roi = _parse_roi(args[i+=1])
        elseif a == "--window"; window = parse(Float64, args[i+=1])
        elseif a == "--hop"; hop = parse(Float64, args[i+=1])
        elseif a == "--fs"; fs = parse(Float64, args[i+=1])
        elseif startswith(a, "-"); error("unknown option $a\n" * CLI_USAGE)
        else path = a
        end
        i += 1
    end
    path === nothing && error("no clip given\n" * CLI_USAGE)
    hr, trace, mon = analyze_clip(path; method=method, roi=roi, fs=fs, window=window, hop=hop, csv=csv)
    ok = isfinite.(hr.bpm)
    println("\n== $(basename(path)) ==")
    println("windows       : $(length(hr.t)) ($(count(ok)) with a valid face)")
    if any(ok)
        b = hr.bpm[ok]; s = hr.snr[ok]
        println("heart rate    : median $(round(median(b), digits=1)) bpm  (IQR $(round(quantile_(b, 0.25), digits=1))–$(round(quantile_(b, 0.75), digits=1)))")
        println("SNR           : median $(round(median(s), digits=1)) dB")
    end
    println("frames        : $(length(trace)), dropped ≈ $(mon.ndropped) in $(mon.ndrop_events) gaps")
    return 0
end

quantile_(x, p) = (s = sort(x); s[clamp(round(Int, 1 + p * (length(s) - 1)), 1, length(s))])
