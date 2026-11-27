# Benchmark the rPPG methods against contact-PPG ground truth.
#
#   julia --project scripts/benchmark.jl ubfc2 [data/datasets/ubfc2]   # UBFC-rPPG dataset 2
#   julia --project scripts/benchmark.jl pure  [data/datasets/pure]    # PURE
#   julia --project scripts/benchmark.jl synthetic                      # generated clips (harness check)
#   options: --methods green,chrom,pos  --snr-min -3  --limit N  --out results
#
# Results: results/benchmark_<dataset>.csv (one row per recording x method) and a summary
# table in results/README.md. ROI traces are cached in results/cache/ (delete to recompute).
using CamHealth

function main(args)
    isempty(args) && (println("usage: benchmark.jl ubfc2|pure|synthetic [root] [--methods ..] [--snr-min x] [--limit N] [--out dir]"); return 1)
    ds = args[1]; root = nothing; methods = collect(RPPG_METHODS); snr_min = -3.0; limit = typemax(Int); out = "results"
    i = 2
    while i <= length(args)
        a = args[i]
        if a == "--methods"; methods = Symbol.(split(args[i+=1], ','))
        elseif a == "--snr-min"; snr_min = parse(Float64, args[i+=1])
        elseif a == "--limit"; limit = parse(Int, args[i+=1])
        elseif a == "--out"; out = args[i+=1]
        else root = a
        end
        i += 1
    end
    roi = :yunet
    if ds == "ubfc2"
        samples = load_ubfc2(something(root, joinpath("data", "datasets", "ubfc2")))
    elseif ds == "pure"
        samples = load_pure(something(root, joinpath("data", "datasets", "pure")))
    elseif ds == "synthetic"
        root = something(root, joinpath("data", "datasets", "synthetic"))
        # 4 subjects, 30 s, pulse + in-band intensity flicker + specular highlights
        roi = write_fake_ubfc(root; n=4, seconds=30, flicker=0.01, specular=0.01)
        rm(joinpath(out, "cache"); recursive=true, force=true)
        samples = load_ubfc2(root)
    else
        error("unknown dataset $ds")
    end
    samples = samples[1:min(limit, length(samples))]
    println("$(length(samples)) recordings from $ds, methods $(join(methods, ", "))")
    _, summary = run_benchmark(samples; methods=Tuple(methods), roi=roi, snr_min=snr_min, outdir=out, tag=ds)
    println()
    for s in summary
        p = s.pooled
        println(rpad(string(s.method), 6), " MAE $(round(p.mae, digits=2)) bpm, RMSE $(round(p.rmse, digits=2)), r $(round(p.r, digits=3)), ",
                "coverage $(round(100s.coverage, digits=1)) %, IBI err $(round(s.ibi_err, digits=1)) ms, RMSSD err $(round(s.rmssd_err, digits=1)) ms")
    end
    return 0
end

exit(main(ARGS))
