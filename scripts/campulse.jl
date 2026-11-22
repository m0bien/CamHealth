# Offline heart rate from a video clip.
#   julia --project scripts/campulse.jl data/clip.mkv --method green --csv data/clip_hr.csv
using CamHealth
exit(CamHealth.cli_main(ARGS))
