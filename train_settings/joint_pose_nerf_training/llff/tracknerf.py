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

from source.utils.config_utils import load_options, save_options_file, override_options, dict_compare
from train_settings.default_config import get_joint_pose_nerf_default_config_llff


def get_config():
    default_config = get_joint_pose_nerf_default_config_llff()
    
    settings_model = edict()

    # camera options    
    settings_model.camera = edict()
    settings_model.camera.initial_pose = 'identity'

    # scheduling
    settings_model.first_joint_pose_nerf_then_nerf = True
    settings_model.ratio_end_joint_nerf_pose_refinement = 0.3
    settings_model.barf_c2f =  [0.4, 0.7]  # 1
    settings_model.start_iter = edict()
    settings_model.start_iter.corres = 1000
    settings_model.start_iter.depth_cons = 1000

    # dataset
    settings_model.dataset = 'llff'
    settings_model.resize = None
    settings_model.llff_img_factor = 8
    
    # flow stuff
    settings_model.use_flow = True
    settings_model.flow_backbone='PDCNet' 

    # loss type
    settings_model.loss_type = 'photometric_and_track_and_depth_reg'
    settings_model.matching_pair_generation = 'all_to_all'

    settings_model.loss_weight = edict()                                               
    settings_model.loss_weight.render = 0.    
    settings_model.loss_weight.corres = -3. 
    
    
    
    settings_model.track=edict()
    settings_model.track.precompute_ka_for_track_loss=True
    settings_model.track.precompute_ka_for_corrs_loss=True
    settings_model.track.enable_ka_for_track_loss=True
    settings_model.track.enable_ka_for_corrs_loss=True
    settings_model.track.mode='mv'
    settings_model.loss_weight.tracks=-3.5 # for 10^(-3)*loss
    settings_model.loss_weight.render_tracks = 0.
    settings_model.track.graph='single' # added
    
    settings_model.max_iter_=60000
    
    settings_model.depth_reg=edict()
    settings_model.depth_reg.gradient_scaling=False # it doesn't work
    settings_model.loss_weight.depth_reg=-1. # 0.1*depth_loss
    
    
    settings_model.depth_reg.start_iter_ratio=settings_model.ratio_end_joint_nerf_pose_refinement
    settings_model.depth_reg.patch_num=1
    settings_model.depth_reg.patch_size=32
    settings_model.depth_reg.depth_regu_patch_size=32
    settings_model.depth_reg.color='gt'
    settings_model.loss_weight.depth_cons = -3.
    
    
    return override_options(default_config, settings_model)

