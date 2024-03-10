"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import time
from pathlib import Path
from easydict import EasyDict as edict
import os

from source.utils.config_utils import override_options
from train_settings.default_config import get_nerf_default_config_360_data


def get_config():
    default_config = get_nerf_default_config_360_data()
    
    settings_model = edict()

    # dataset
    settings_model.dataset = 'dtu'
    settings_model.resize = None

    # scheduling with coarse to fine 
    settings_model.barf_c2f =  [0.1, 0.5]  # coarse-to-fine pe

    # fine sampling
    settings_model.nerf = edict()                                                       # NeRF-specific options
    settings_model.nerf.depth = edict()                                                  # depth-related options
    settings_model.nerf.depth.param = 'metric'                                       # depth parametrization (for sampling along the ray)
    settings_model.nerf.fine_sampling = True                                  # hierarchical sampling with another NeRF

    # flow stuff
    settings_model.use_flow = True
    settings_model.flow_backbone='PDCNet'
    settings_model.filter_corr_w_cc = True  
    # leads to slightly better results to have additional filtering on the correspondences here

    # loss type
    settings_model.loss_type = 'photometric_and_track_and_depth_reg' 
    settings_model.matching_pair_generation = 'all_to_all'

    settings_model.gradually_decrease_corres_weight = True 

    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0.    
    settings_model.loss_weight.corres = -4  # for 10^
    settings_model.loss_weight.depth_cons = -3
    
    settings_model.track=edict()
    settings_model.track.precompute_ka_for_track_loss=True
    settings_model.track.precompute_ka_for_corrs_loss=True
    settings_model.track.enable_ka_for_track_loss=True
    settings_model.track.enable_ka_for_corrs_loss=True
    settings_model.track.mode='mv'
    settings_model.track.graph='single' # added
    settings_model.loss_weight.tracks=-3.5
    settings_model.loss_weight.render_tracks = 0.
    

    settings_model.max_iter_=50000
    
    settings_model.depth_reg=edict()
    settings_model.depth_reg.gradient_scaling=False # it doesn't work
    settings_model.loss_weight.depth_reg=-1.
    # terminal 3 run -1, terminal 4 run -2
    
    # this is for color and depth gradient based smooth
    # settings_model.depth_reg.start_iter_ratio=settings_model.ratio_end_joint_nerf_pose_refinement
    settings_model.depth_reg.patch_num=1
    settings_model.depth_reg.patch_size=32
    settings_model.depth_reg.depth_regu_patch_size=32
    settings_model.depth_reg.color='gt'
    
    
    return override_options(default_config, settings_model)

