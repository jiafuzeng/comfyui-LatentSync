#!/bin/bash

python -m scripts.inference \
    --unet_config_path "configs/unet/stage2.yaml" \
    --inference_ckpt_path "checkpoints/latentsync_unet.pt" \
    --inference_steps 20 \
    --guidance_scale 1.5 \
    --video_path "assets/5min视频.mp4" \
    --audio_path "assets/5min音频.MP3" \
    --video_out_path "video_out_5min.mp4"
