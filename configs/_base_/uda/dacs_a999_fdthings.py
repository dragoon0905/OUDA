# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# UDA with Thing-Class ImageNet Feature Distance + Increased Alpha
_base_ = ['dacs.py']
uda = dict(
    alpha=0.999,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[5,9,10,11,12], #[6, 7, 11, 12, 13, 14, 15, 16, 17, 18], # modified
    imnet_feature_dist_scale_min_ratio=0.75,
)
