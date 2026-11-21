# M0 camera spike: open the webcam with VideoIO, grab ~150 frames, report
# resolution, measured fps (from timestamps) and pixel format.
#   julia --project scripts/camera_check.jl                 # C930e, 640x480
#   julia --project scripts/camera_check.jl "video=Logitech Webcam C930e" 640x480
using VideoIO, Statistics

device = length(ARGS) >= 1 ? ARGS[1] : "video=Logitech Webcam C930e"
size_s = length(ARGS) >= 2 ? ARGS[2] : "640x480"
nframes = 150

# YUY2 (uncompressed) 640x480@30 is the target mode for colour fidelity (see PLAN.md)
fmt = VideoIO.libffmpeg.av_find_input_format("dshow")
opts() = VideoIO.AVDict("video_size" => size_s, "framerate" => "30", "pixel_format" => "yuyv422")
println("Opening $device (dshow, $size_s @ 30 fps, yuyv422) ...")
cam = try
    VideoIO.opencamera(device, fmt, opts())
catch err
    println("opencamera with explicit mode failed: ", sprint(showerror, err))
    println("Retrying with VideoIO defaults ...")
    VideoIO.opencamera(device)
end

function pixfmt_info(r)
    try
        v = r.input_pix_fmt
        "camera/decoder $(unsafe_string(VideoIO.libffmpeg.av_get_pix_fmt_name(v))) -> Julia RGB{N0f8}"
    catch err
        "n/a ($(typeof(err)))"
    end
end

pixfmt = pixfmt_info(cam)
frame = read(cam)                      # first frame (warm-up)
ts = Float64[]; pts = Float64[]
t0 = time()
for i in 1:nframes
    read!(cam, frame)
    push!(ts, time() - t0)
    push!(pts, try Float64(VideoIO.gettime(cam)) catch; NaN end)
end
close(cam)

dt = diff(ts)
println("\n== Camera check ==")
println("device        : $device")
println("frame size    : $(size(frame, 2))x$(size(frame, 1)) (W x H), eltype $(eltype(frame))")
println("pixel format  : ", pixfmt)
println("frames        : $nframes in $(round(ts[end], digits=2)) s")
println("fps (wall)    : $(round((nframes - 1) / (ts[end] - ts[1]), digits=2))")
if all(isfinite, pts) && pts[end] > pts[1]
    println("fps (pts)     : $(round((nframes - 1) / (pts[end] - pts[1]), digits=2))")
end
println("frame interval: mean $(round(1000mean(dt), digits=1)) ms, std $(round(1000std(dt), digits=1)) ms, max $(round(1000maximum(dt), digits=1)) ms")
m = mean(Float64.(VideoIO.red.(frame))), mean(Float64.(VideoIO.green.(frame))), mean(Float64.(VideoIO.blue.(frame)))
println("mean RGB last : ", round.(m, digits=3))
fps = (nframes - 1) / (ts[end] - ts[1])
if fps < 25
    println("\nNOTE: below 30 fps. With auto-exposure on, the C930e lengthens exposure in dim light and")
    println("drops the frame rate. Add light and/or lock exposure (see README) before recording.")
end
