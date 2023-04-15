#!/bin/bash # For syntax highlighting :}
# (this is . sourced by other scripts)

# This $0 references the CALLER SCRIPT's name
ourname="${0##*/}"

# Other settings:
ourutilsdir="$ourdir/utils"
ourdatadir="$ourdir/data"
bin_vid="$ourutilsdir/video-to-frames"
bin_blurdet="$ourutilsdir/blur-detect.py"

dir_origframes='1-frames-raw'           # Extracted frames
dir_data='gendata'                      # Generated data
dir_blur_approve='2-frames-nonblurry'   # Symlinks after approval
dir_subset='3-frames-subset'            # Symlinks subset
dir_pg='6-pg-for-reconstruction'        # Created images (tif?) for 3d photog...

fn_blur="$dir_data/blur.txt"
fn_exif_tags_settable="$ourdatadir/exif-img-tags-settable"
fn_exif_data="$dir_data/exif-img-tags.txt"

if [[ "$*" == *"-v"* ]]; then
	echo "This script name    : $ourname"
	echo "Pipeline scripts dir: $ourdir"
	echo "Pipeline utils dir  : $ourname"
fi

s0 () { stdbuf -i0 -o0 "$@"; }

# User settings:
yn_prompt_timeout=3
