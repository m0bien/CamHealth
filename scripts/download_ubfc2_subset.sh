#!/bin/sh
# Research use only; cite Bobbia et al. 2017. Do not redistribute. ~7.6 GB.
# UBFC-rPPG DATASET_2, subjects 1 3 4 5 8 (public Google Drive folder from the official page)
cd "$(dirname "$0")/../data/datasets/ubfc2" 2>/dev/null || { mkdir -p "$(dirname "$0")/../data/datasets/ubfc2"; cd "$(dirname "$0")/../data/datasets/ubfc2"; }
get() { mkdir -p "$1"; curl -sSL --retry 4 -o "$1/$2" "https://drive.usercontent.google.com/download?id=$3&export=download&confirm=t"; echo "$1/$2 $(wc -c < "$1/$2")"; }
get subject1 ground_truth.txt 1q7LX_8Ggfl43ZmsFXMrUkV2hNnKr-yTE
get subject1 vid.avi          1qBlkbaB8y3-KlWC_A61KsY42Wjo66ss4
get subject3 ground_truth.txt 1tKRzecjw14TGFkizo4XvMEWXeefriiCH
get subject3 vid.avi          1tL5EX50qFD8n6VO0wX9x_iOrPKyVR2-W
get subject4 ground_truth.txt 1vvyDn_3hzw-AvgX69Ct3O2omH_D_erHj
get subject4 vid.avi          1w00W_8bNiKKLvZ-p3n3iESLG_gr9AOoz
get subject5 ground_truth.txt 1yD3ng0b2D5gFdFMfmhOphCjunXeuU4IB
get subject5 vid.avi          1yGaYtOAGYzI4UOIN-XArEl88eRjRarTc
get subject8 ground_truth.txt 1yKqVlkTkBp93KXUvmniOZ3JVfK-HxNnP
get subject8 vid.avi          1yLAa1v0tdL3TZM4CEXkbslI0qyiqyRkH
echo DONE
