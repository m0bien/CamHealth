# Download the YuNet face detector (OpenCV Zoo, MIT licence) into data/models/.
#   julia --project scripts/download_models.jl
using Downloads, SHA
const URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
dir = joinpath(@__DIR__, "..", "data", "models")
mkpath(dir)
dest = joinpath(dir, "face_detection_yunet_2023mar.onnx")
if isfile(dest) && filesize(dest) > 100_000
    println("already present: $dest")
else
    println("downloading $URL")
    Downloads.download(URL, dest)
end
println("size   : $(filesize(dest)) bytes")
println("sha256 : $(bytes2hex(open(sha256, dest)))")
filesize(dest) > 100_000 || error("download looks wrong (Git LFS pointer?): $(filesize(dest)) bytes")
