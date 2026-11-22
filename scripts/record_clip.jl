# Record a lossless clip from the webcam for offline analysis.
#
#   julia --project scripts/record_clip.jl [seconds=30] [out=data/clip_<time>.mkv] [--no-lock] [--keep-lock] [--device NAME]
#
# 1. warm-up: stream for a few seconds so auto exposure / white balance / focus settle,
#    then lock them (scripts/camera_lock.ps1) while the camera is still streaming;
# 2. record YUY2 640x480 @ 30 fps with FFmpeg into FFV1 (lossless, yuv422p — the camera's
#    native chroma, no colour loss) in Matroska, using the device timestamps;
# 3. read the clip back and write per-frame timestamps to <out>_frames.csv
#    (frame, t_s, dt_ms, dropped_before) plus a drop / brightness-drift summary.
# Afterwards the camera is set back to auto (use --keep-lock to leave it locked).
using CamHealth, VideoIO, Dates, Statistics

function main(args)
    secs = 30.0; out = nothing; lock = true; keep = false; device = "Logitech Webcam C930e"; warm = 5.0
    pos = String[]; i = 1
    while i <= length(args)
        a = args[i]
        if a == "--no-lock"; lock = false
        elseif a == "--keep-lock"; keep = true
        elseif a == "--device"; device = args[i+=1]
        elseif a == "--warmup"; warm = parse(Float64, args[i+=1])
        else push!(pos, a)
        end
        i += 1
    end
    length(pos) >= 1 && (secs = parse(Float64, pos[1]))
    datadir = joinpath(@__DIR__, "..", "data"); mkpath(datadir)
    out = length(pos) >= 2 ? pos[2] : joinpath(datadir, "clip_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".mkv")
    ff = VideoIO.FFMPEG_jll.ffmpeg()
    input = `-f dshow -video_size 640x480 -framerate 30 -pixel_format yuyv422 -rtbufsize 512M -i video=$device`

    # --- warm-up + lock ---
    println("warm-up $(warm) s (auto exposure settling)...")
    p = run(pipeline(`$ff -hide_banner -loglevel error $input -t $warm -f null -`, stdin=devnull); wait=false)
    sleep(max(warm - 1.5, 1.0))
    if lock
        try
            print(CamHealth.lock_camera(:lock; device=device))
        catch err
            @warn "camera lock failed — recording with auto settings" err
        end
    end
    wait(p)

    # --- record ---
    println("recording $(secs) s -> $out")
    log = out * ".ffmpeg.log"
    cmd = `$ff -hide_banner -y $input -t $secs -c:v ffv1 -level 3 -pix_fmt yuv422p -g 1 $out`
    try
        run(pipeline(cmd, stdin=devnull, stderr=log))
    finally
        # give other apps (video calls) their auto settings back unless asked not to
        lock && !keep && try print(CamHealth.lock_camera(:auto; device=device)) catch end
    end

    # --- timestamps + stats ---
    mon = CamHealth.FrameMonitor(nominal_fps=30, warn=false)
    src = FileSource(out; monitor=mon)
    for _ in src end
    close(src)
    t = mon.times; dt = [NaN; diff(t)]
    csv = replace(out, r"\.mkv$" => "") * "_frames.csv"
    open(csv, "w") do io
        println(io, "frame,t_s,dt_ms,dropped_before")
        for k in eachindex(t)
            println(io, k, ",", round(t[k], digits=4), ",", round(1000dt[k], digits=2), ",", Int(k > 1 && dt[k] > 1.5 / 30))
        end
    end
    b = mon.brightness
    println("\n== clip ==")
    println("file          : $out ($(round(filesize(out) / 2^20, digits=1)) MiB)")
    println("frames        : $(length(t)) in $(round(t[end], digits=2)) s -> $(round((length(t) - 1) / t[end], digits=2)) fps")
    println("frame interval: mean $(round(1000mean(dt[2:end]), digits=1)) ms, max $(round(1000maximum(dt[2:end]), digits=1)) ms")
    println("drops         : ≈$(mon.ndropped) frames missing in $(mon.ndrop_events) gaps (interval > 1.5x nominal)")
    println("brightness    : mean $(round(mean(b), digits=3)), range $(round(minimum(b), digits=3))–$(round(maximum(b), digits=3)), $(mon.ndrift) samples drifted >5% within 2 s")
    println("timestamps    : $csv")
    println("ffmpeg log    : $log")
end

main(ARGS)
