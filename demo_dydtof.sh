# Copyright (c) Meta Platforms, Inc. and affiliates.

python video_demo.py configs/dvsr_config.py chkpts/dvsr_tartan.pth data/demo_dydtof results/demo_dydtof_dvsr --device 0
python video_demo.py configs/hvsr_config.py chkpts/hvsr_tartan.pth data/demo_dydtof results/demo_dydtof_hvsr --device 0
