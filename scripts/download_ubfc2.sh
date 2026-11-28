#!/bin/sh
# Download UBFC-rPPG DATASET_2 into data/datasets/ubfc2/ (research use only, cite Bobbia et al. 2017,
# do not redistribute). ~75 GB for all 42 subjects (1.26-1.87 GB per video).
# Resumable: complete files (size matches the server) are skipped, partial files are resumed.
# Google Drive limits downloads per file per day; when it answers "Quota exceeded" the script
# stops (nothing is written) - run it again later.
#   sh scripts/download_ubfc2.sh                 # everything
#   sh scripts/download_ubfc2.sh --sizes         # only print remote sizes
#   sh scripts/download_ubfc2.sh subject1 subject3
here=$(cd "$(dirname "$0")" && pwd)
dest="$here/../data/datasets/ubfc2"; mkdir -p "$dest"
CR=$(printf '\r')
url() { echo "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"; }
# headers of a 1-byte range request: size from Content-Range, and the content type
probe() { curl -sL -r 0-0 -D - -o /dev/null "$(url "$1")" | tr -d "$CR"; }
lsize() { if [ -f "$1" ]; then wc -c < "$1" | tr -d ' '; else echo 0; fi; }
sizes_only=0; [ "$1" = "--sizes" ] && { sizes_only=1; shift; }
want="$*"
grep -v '^#' "$here/ubfc2_manifest.txt" | while read -r subj gt vid; do
  [ -z "$subj" ] && continue
  [ -n "$want" ] && ! echo " $want " | grep -q " $subj " && continue
  for f in "ground_truth.txt $gt" "vid.avi $vid"; do
    set -- $f; name=$1; id=$2; out="$dest/$subj/$name"
    hdr=$(probe "$id")
    rs=$(echo "$hdr" | awk 'tolower($1)=="content-range:"{split($3,a,"/"); n=a[2]} END{print n+0}')
    if echo "$hdr" | grep -qi '^content-type: text/html'; then
      echo "QUOTA $subj/$name: Google Drive refused the download (quota exceeded) - try again later"
      exit 3
    fi
    if [ "$sizes_only" = 1 ]; then echo "$subj/$name $rs"; continue; fi
    # never resume a video that does not start like an AVI (an earlier error page would stay glued in front)
    if [ "$name" = vid.avi ] && [ -f "$out" ] && [ "$(head -c 4 "$out")" != RIFF ]; then
      echo "bad   $subj/$name: not an AVI, starting over"; rm -f "$out"
    fi
    ls_=$(lsize "$out")
    if [ "$rs" -gt 0 ] && [ "$ls_" = "$rs" ]; then echo "ok    $subj/$name ($rs)"; continue; fi
    mkdir -p "$dest/$subj"
    curl -sSL --retry 5 --retry-delay 5 -C - -o "$out" "$(url "$id")"
    ls_=$(lsize "$out")
    if [ "$name" = vid.avi ] && [ -f "$out" ] && [ "$(head -c 4 "$out")" != RIFF ]; then
      rm -f "$out"; echo "QUOTA $subj/$name - got an error page, removed"; exit 3
    fi
    if [ "$ls_" = "$rs" ]; then
      echo "done  $subj/$name ($rs)"
    else
      # an HTML error page instead of the file: remove it so a later run starts clean
      if [ "$ls_" -lt 100000 ] && grep -qi '<html' "$out" 2>/dev/null; then rm -f "$out"; echo "QUOTA $subj/$name - removed error page"; exit 3; fi
      echo "FAIL  $subj/$name local $ls_ != remote $rs"
    fi
  done
done
status=$?
[ $status -eq 0 ] && echo FINISHED
exit $status
