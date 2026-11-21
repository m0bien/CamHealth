# List DirectShow devices and the C930e's capture modes with VideoIO's bundled ffmpeg.
#   julia --project scripts/list_camera_modes.jl ["Logitech Webcam C930e"]
using VideoIO
const FF = VideoIO.FFMPEG_jll
device = isempty(ARGS) ? "Logitech Webcam C930e" : ARGS[1]

function runff(args; timeout=30)
    cmd = `$(FF.ffmpeg()) -hide_banner $args`        # JLL Cmd sets PATH for its DLLs
    p = run(pipeline(ignorestatus(cmd), stdin=devnull, stdout=stdout, stderr=stdout); wait=false)
    t0 = time()
    while process_running(p) && time() - t0 < timeout
        sleep(0.1)
    end
    process_running(p) && (kill(p); println("[timed out after $(timeout)s]"))
end

println("== dshow devices =="); runff(`-list_devices true -f dshow -i dummy`)
println("\n== modes of $device =="); runff(`-f dshow -list_options true -i video=$device`)
