# Benchmark datasets with contact-PPG ground truth.
#
#   UBFC-rPPG dataset 2  (Bobbia et al. 2017): <root>/subjectN/vid.avi + ground_truth.txt
#       ground_truth.txt: row 1 PPG waveform, row 2 PPG heart rate, row 3 time (s)
#   PURE (Stricker et al. 2014): <root>/NN-MM/Image<ns>.png + <root>/NN-MM.json
#       json: "/FullPackage" = [{Timestamp (ns), Value: {waveform, pulseRate, ...}}],
#             "/Image" = [{Timestamp (ns)}]
#
# Neither dataset may be redistributed; keep them under data/datasets/ (git-ignored).

import JSON
import Random
using FileIO: load
using VideoIO: N0f8
using Statistics: mean, median, std

"""One benchmark recording: a way to open its frames, and the reference PPG."""
struct BenchSample
    dataset::String
    name::String
    open_frames::Function        # () -> frame source iterating (frame, t)
    ppg_t::Vector{Float64}       # reference PPG time (s), relative to the first frame
    ppg::Vector{Float64}         # reference PPG waveform (systole = maximum)
end

Base.show(io::IO, s::BenchSample) =
    print(io, "BenchSample($(s.dataset)/$(s.name), $(length(s.ppg)) PPG samples, $(round(s.ppg_t[end] - s.ppg_t[1], digits=1)) s)")

# ---------------------------------------------------------------------------
# UBFC-rPPG dataset 2
# ---------------------------------------------------------------------------

"""Read UBFC-rPPG dataset 2's `ground_truth.txt` -> (t_s, ppg, hr)."""
function read_ubfc2_gt(path::AbstractString)
    rows = [parse.(Float64, split(strip(l))) for l in eachline(path) if !isempty(strip(l))]
    length(rows) >= 3 || error("$path: expected 3 rows (PPG, HR, time), got $(length(rows))")
    ppg, hr, t = rows[1], rows[2], rows[3]
    length(ppg) == length(t) || error("$path: PPG and time rows differ in length")
    # time is in seconds in dataset 2; be tolerant of millisecond files
    median(diff(t)) > 1 && (t = t ./ 1000)
    return t .- t[1], ppg, hr
end

"""
    load_ubfc2(root) -> Vector{BenchSample}

Every `subject*/` folder under `root` with `vid.avi` and `ground_truth.txt`, sorted by
subject number.
"""
function load_ubfc2(root::AbstractString)
    isdir(root) || error("UBFC-rPPG folder not found: $root")
    dirs = filter(d -> isfile(joinpath(root, d, "vid.avi")) && isfile(joinpath(root, d, "ground_truth.txt")), readdir(root))
    num(d) = (m = match(r"(\d+)", d); m === nothing ? typemax(Int) : parse(Int, m[1]))
    sort!(dirs; by=num)
    map(dirs) do d
        t, ppg, _ = read_ubfc2_gt(joinpath(root, d, "ground_truth.txt"))
        vid = joinpath(root, d, "vid.avi")
        BenchSample("ubfc2", d, () -> FileSource(vid), t, ppg)
    end
end

# ---------------------------------------------------------------------------
# PURE
# ---------------------------------------------------------------------------

"""Frame source over an image sequence with given timestamps (s)."""
mutable struct ImageSeqSource <: AbstractFrameSource
    paths::Vector{String}
    times::Vector{Float64}
end
function Base.iterate(src::ImageSeqSource, k=1)
    k > length(src.paths) && return nothing
    img = load(src.paths[k])
    return ((RGB{N0f8}.(img), src.times[k] - src.times[1]), k + 1)
end
Base.close(::ImageSeqSource) = nothing

"""Read a PURE json -> (ppg_t_s, ppg, image_t_s), times relative to the first image."""
function read_pure_json(path::AbstractString)
    j = JSON.parsefile(path)
    pk = j["/FullPackage"]; im = j["/Image"]
    tp = [Float64(p["Timestamp"]) for p in pk] ./ 1e9
    ppg = [Float64(p["Value"]["waveform"]) for p in pk]
    ti = [Float64(p["Timestamp"]) for p in im] ./ 1e9
    t0 = ti[1]
    return tp .- t0, ppg, ti .- t0
end

"""
    load_pure(root) -> Vector{BenchSample}

Every `NN-MM/` folder of PNG frames with a matching `NN-MM.json` (next to the folder or
inside it).
"""
function load_pure(root::AbstractString)
    isdir(root) || error("PURE folder not found: $root")
    out = BenchSample[]
    for d in sort(filter(d -> isdir(joinpath(root, d)) && occursin(r"^\d+-\d+$", d), readdir(root)))
        js = [joinpath(root, d * ".json"), joinpath(root, d, d * ".json")]
        k = findfirst(isfile, js)
        k === nothing && continue
        tp, ppg, ti = read_pure_json(js[k])
        pngs = sort(filter(f -> endswith(lowercase(f), ".png"), readdir(joinpath(root, d))))
        ts = [parse(Float64, match(r"(\d+)", f)[1]) / 1e9 for f in pngs]
        paths = joinpath.(root, d, pngs)
        o = sortperm(ts)
        tsrc = ts[o] .- ts[o][1]
        push!(out, BenchSample("pure", d, () -> ImageSeqSource(paths[o], tsrc .+ ts[o][1]), tp, ppg))
    end
    return out
end

# ---------------------------------------------------------------------------
# Synthetic dataset (harness validation and tests)
# ---------------------------------------------------------------------------

"""
    synthetic_ppg(t; hr_bpm, hrv_ms=40, seed) -> (ppg, beat_times)

Smooth PPG-like waveform (systolic Gaussian + dicrotic bump) with beat-to-beat variation.
"""
function synthetic_ppg(t::AbstractVector{<:Real}; hr_bpm::Real=72, hrv_ms::Real=40, seed::Integer=1)
    rng = Random.MersenneTwister(seed)
    rr = 60 / hr_bpm
    beats = Float64[0.3]
    while beats[end] < t[end] + 1
        push!(beats, beats[end] + rr + hrv_ms / 1000 * randn(rng))
    end
    p = zeros(length(t))
    for b in beats
        @. p += exp(-((t - b) / 0.09)^2 / 2) + 0.35 * exp(-((t - b - 0.3) / 0.08)^2 / 2)
    end
    return p, beats
end

"""
    write_fake_ubfc(root; n=2, seconds=20, fs=30, sz=(48, 64), roi=(15:30, 21:40), hr=[...],
                    amp=0.006, flicker=0.0, specular=0.0)

Write a tiny UBFC-rPPG-shaped dataset: `subjectK/vid.avi` (FFV1 in AVI) and
`ground_truth.txt`. Inside `roi` the skin colour is modulated along the blood-absorption
direction by a known PPG; `flicker` adds a common-mode intensity change and `specular`
a white specular term (both at 1.7 Hz, in the pulse band). Returns the ROI as `FixedROI`.
"""
function write_fake_ubfc(root::AbstractString; n::Integer=2, seconds::Real=20, fs::Real=30, sz=(48, 64),
                         roi=(15:30, 21:40), hr=[66.0, 84.0, 75.0, 90.0], amp::Real=0.006,
                         flicker::Real=0.0, specular::Real=0.0, seed::Integer=7)
    rng = Random.MersenneTwister(seed)
    skin = (0.70, 0.50, 0.40)
    pulsedir = (0.33, 0.77, 0.53)
    t = collect(0:1/fs:seconds-1/fs)
    for s in 1:n
        dir = joinpath(root, "subject$s"); mkpath(dir)
        ppg, _ = synthetic_ppg(t; hr_bpm=hr[mod1(s, length(hr))], seed=seed + s)
        pz = (ppg .- mean(ppg)) ./ std(ppg)
        frames = Matrix{RGB{N0f8}}[]
        for (k, τ) in enumerate(t)
            I = 1 + flicker * sin(2π * 1.7 * τ)
            spec = specular * (1 + sin(2π * 1.7 * τ + 1))
            f = Array{RGB{N0f8}}(undef, sz...)
            for j in 1:sz[2], i in 1:sz[1]
                inroi = i in roi[1] && j in roi[2]
                c = ntuple(ch -> skin[ch] * (inroi ? 1 - amp * pulsedir[ch] * pz[k] : 0.8), 3)
                v = ntuple(ch -> clamp(I * (c[ch] + spec) + 0.01 * randn(rng), 0, 1), 3)
                f[i, j] = RGB{N0f8}(v...)
            end
            push!(frames, f)
        end
        VideoIO.save(joinpath(dir, "vid.avi"), frames; framerate=fs, codec_name="ffv1",
                     target_pix_fmt=VideoIO.AV_PIX_FMT_BGR0)
        hrrow = fill(hr[mod1(s, length(hr))], length(t))
        open(joinpath(dir, "ground_truth.txt"), "w") do io
            println(io, join(round.(ppg, digits=5), "  "))
            println(io, join(hrrow, "  "))
            println(io, join(round.(t, digits=5), "  "))
        end
    end
    return FixedROI(roi[2].start - 1, roi[1].start - 1, length(roi[2]), length(roi[1]))
end
