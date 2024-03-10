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

import numpy as np
import torch
from easydict import EasyDict as edict
from typing import Any, Dict, Tuple
from source.training.core.base_losses import BaseLoss
from source.training.core.correspondence_utils import (generate_pair_list, 
                                                       image_pair_candidates_with_angular_distance, 
                                                       CorrrespondenceUtils, get_mask_valid_from_conf_map)
from source.utils.config_utils import override_options
from source.utils.colmap_initialization.pdcnet_for_hloc import get_grid_keypoints
import h5py
class CorrespondenceBasedLoss(BaseLoss, CorrrespondenceUtils):
    """Correspondence Loss. Main signal for the joint pose-NeRF training. """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(device=device)
        default_cfg = edict({'matching_pair_generation': 'all', 
                             'min_nbr_matches': 500, 
                            
                             'pairing_angle_threshold': 30, # degree, in case 'angle' pair selection chosen
                             'filter_corr_w_cc': False, 
                             'min_conf_valid_corr': 0.95, 
                             'min_conf_cc_valid_corr': 1./(1. + 1.5), 
                             })
        self.opt = override_options(default_cfg, opt)
        self.device = device

        self.train_data = train_data
        H, W = train_data.all.image.shape[-2:]
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        self.grid = torch.stack((xx, yy), dim=-1).to(self.device).float()  # ( H, W, 2)
        self.grid_flat = self.grid[:, :, 1] * W + self.grid[:, :, 0]  # (H, W), corresponds to index in flattedned array (in H*W)
        self.grid_flat = self.grid_flat.to(self.device).long()


        self.net = nerf_net
        self.flow_net = flow_net

        self.gt_corres_map_and_mask_all_to_all = None
        if 'depth_gt' in train_data.all.keys():
            self.gt_corres_map_and_mask_all_to_all = self.get_gt_correspondence_maps_all_to_all(n_views=len(train_data))
            # (N, N, 3, H, W)

        self.compute_correspondences(train_data) # * note : the 2d-2d matching is computed from here
        # flow will either be computed on the fly (if weights are finetuned or not)

    @torch.no_grad()
    def compute_correspondences(self, train_data: Dict[str, Any]):
        """Compute correspondences relating the input views. 

        Args:
            train_data (dataset): training dataset. The keys all is a dictionary, 
                                  containing the entire training data. 
                                  train_data.all has keys 'idx', 'image', 'intr', 'pose' 
                                  and all images of the scene already stacked here.

        """
       
        
        print('Computing flows')
        images = train_data.all['image']  # (N_views, 3, H, W)
        H, W = images.shape[-2:]
        poses = train_data.all['pose']  # ground-truth poses w2c
        n_views = images.shape[0]

        if self.opt.matching_pair_generation == 'all':
            # exhaustive pairs, but (1, 2) or (2, 1) are treated as the same. 
            combi_list = generate_pair_list(n_views)
        elif self.opt.matching_pair_generation == 'all_to_all': # * default behavior
            # all pairs, including both directions. (1, 2) and (2, 1)
            combi_list = self.flow_net.combi_list  # 2xN
            # first row is target, second row is source
        elif self.opt.matching_pair_generation == 'angle':
            # pairs such as the angular distance between images is below a certain threshold
            combi_list = image_pair_candidates_with_angular_distance\
                (poses, pairing_angle_threshold=self.opt.pairing_angle_threshold)
        else:
            raise ValueError
        # [0, 0, 1, 1, 2, 2],[1, 2, 0, 2, 0, 1] example all_to_all from 3 views
        print(f'Computing {combi_list.shape[1]} correspondence maps')
        if combi_list.shape[1] == 0: 
            self.flow_plot, self.flow_plot_masked = None, None
            self.corres_maps, self.conf_maps, self.mask_valid_corr = None, None, None
            self.filtered_flow_pairs = []
            return 

        # IMPORTANT: the batch norm should be to eval!!
        # otherwise, the statistics of the image are too different and it does something bad!
        if self.opt.filter_corr_w_cc: # * true in replica dataset
            corres_maps, conf_maps, conf_maps_from_cc, flow_plot = self.flow_net.compute_flow_and_confidence_map_and_cc_of_combi_list\
                (images, combi_list_tar_src=combi_list, plot=True, 
                use_homography=self.opt.use_homography_flow) 
        else: # * default here, for conf_maps, default there are 33% of very confident correspondance in [0.9,1]
            corres_maps, conf_maps, flow_plot = self.flow_net.compute_flow_and_confidence_map_of_combi_list\
                (images, combi_list_tar_src=combi_list, plot=True, 
                use_homography=self.opt.use_homography_flow) 
        mask_valid_corr = get_mask_valid_from_conf_map(p_r=conf_maps.reshape(-1, 1, H, W), # about 30% is valid
                                                       corres_map=corres_maps.reshape(-1, 2, H, W), 
                                                       min_confidence=self.opt.min_conf_valid_corr)  # (n_views*(n_views-1), 1, H, W)
        # * 0.95 min confidence for replica
        if self.opt.filter_corr_w_cc:
            mask_valid_corr = mask_valid_corr & conf_maps_from_cc.ge(self.opt.min_conf_cc_valid_corr)
        
        # save the flow examples for tensorboard
        self.flow_plot = flow_plot  
        self.flow_plot_masked = None
        flow_plot = self.flow_net.visualize_mapping_combinations(images=train_data.all.image, 
                                                                 mapping_est=corres_maps.reshape(-1, 2, H, W), 
                                                                 batched_conf_map=mask_valid_corr.float(), 
                                                                 combi_list=combi_list, save_path=None)
        flow_plot = torch.from_numpy( flow_plot.astype(np.float32)/255.).permute(2, 0, 1)
        self.flow_plot_masked = flow_plot

        # when we only computed a subset
        self.corres_maps = corres_maps  # (combi_list.shape[1], 3, H, W), already input rgb size
        self.conf_maps = conf_maps
        self.mask_valid_corr = mask_valid_corr # * note we save the mask here and then use it for sampling
        # should be list of the matching index for each of the image. 
        # first row/element corresponds to the target image, second is the source image
        flow_pairs = (combi_list.cpu().numpy().T).tolist()  
        self.flow_pairs = flow_pairs
        # list of pairs, the target is the first element, source is second
        assert self.corres_maps.shape[0] == len(flow_pairs)

        # keep only the correspondences for which there are sufficient confident regions 
        filtered_flow_pairs = []
        for i in range(len(flow_pairs)):
            nbr_confident_regions = self.mask_valid_corr[i].sum()
            if nbr_confident_regions > self.opt.min_nbr_matches:
                filtered_flow_pairs.append((i, flow_pairs[i][0], flow_pairs[i][1]))
                # corresponds to index_of_flow, index_of_target_image, index_of_source_image
        self.filtered_flow_pairs = filtered_flow_pairs # usually all pairs are valid
        print(f'{len(self.filtered_flow_pairs)} possible flow pairs')
        
        
        # * keypoint adjustment
        if self.opt.track.enable_ka_for_corrs_loss:
            self.refine_keypoint(combi_list, train_data)
        return 

    # THE CORRESPONDENCE LOSSES 
    def sample_valid_image_pair(self):
        """select an image pair in the filtered pair and retrieve corresponding 
        correspondence, confidence map and valid mask. 
        
        Returns: 
            if_self
            id_matching_view
            corres_map_self_to_other_ (H, W, 2)
            conf_map_self_to_other_ (H, W, 1)
            variance_self_to_other_ (H, W, 1) or None
            mask_correct_corr (H, W, 1)
        """
        id_in_flow_list = np.random.randint(len(self.filtered_flow_pairs)) # sample a flow
        # we need to store it for following calculation
        self.sampled_id_in_filtered_flow_list=id_in_flow_list
        id_in_flow_tensor, id_self, id_matching_view = self.filtered_flow_pairs[id_in_flow_list]
        corres_map_self_to_other_ = self.corres_maps[id_in_flow_tensor].permute(1, 2, 0)[:, :, :2]  # (H, W, 2)
        conf_map_self_to_other_ = self.conf_maps[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
        variance_self_to_other_ = None
        mask_correct_corr = self.mask_valid_corr[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)

        return id_self, id_matching_view, corres_map_self_to_other_, \
            conf_map_self_to_other_, variance_self_to_other_, mask_correct_corr

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], 
                     output_dict: Dict[str, Any], iteration: int, mode: str=None, plot: bool=False
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Will not be used here, because rendering must be where 
                                 a match is available. 
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """
        
        if mode != 'train':
            # only during training
            return {}, {}, {}
        
        loss_dict, stats_dict, plotting_dict = self.compute_loss_pairwise\
            (opt, data_dict, output_dict, iteration, mode, plot)
        
        if self.opt.gradually_decrease_corres_weight: #  true for dtu
            # gamma = 0.1**(max(iteration - self.opt.start_iter_photometric, 0)/self.opt.max_iter)
            # reduce the corres weight by 2 every x iterations 
            iter_start_decrease_corres_weight = self.opt.ratio_start_decrease_corres_weight * self.opt.max_iter \
                if self.opt.ratio_start_decrease_corres_weight is not None \
                    else self.opt.iter_start_decrease_corres_weight
            if iteration < iter_start_decrease_corres_weight:
                gamma = 1. 
            else:
                gamma = 2 ** ((iteration-iter_start_decrease_corres_weight) // self.opt.corres_weight_reduct_at_x_iter)
            loss_dict['corres'] = loss_dict['corres'] / gamma
        return loss_dict, stats_dict, plotting_dict

    
    def compute_loss_pairwise(self, opt: Dict[str, Any], data_dict: Dict[str, Any], 
                              output_dict: Dict[str, Any], iteration: int, mode: str=None, plot: bool=False
                              ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Will not be used here, because rendering must be where 
                                 a match is available. 
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """

        stats_dict, plotting_dict, loss_dict = {}, {}, {'corres': torch.tensor(0., requires_grad=True).to(self.device), 
                                                        'render_matches': torch.tensor(0., requires_grad=True).to(self.device)}
        
        if mode != 'train':
            # only during training
            return loss_dict, stats_dict, plotting_dict

        if iteration < self.opt.start_iter.corres: # todo tomorrow consider change it here to check the correspondence loss
            # if the correspondence loss is only added after x iterations
            return loss_dict, stats_dict, plotting_dict

        if len(self.filtered_flow_pairs) == 0:
            return loss_dict, stats_dict, plotting_dict
        # * flow sampling happens here, sample image pairs is enough
        id_self, id_matching_view, corres_map_self_to_other_, conf_map_self_to_other_, variance_self_to_other_, \
            mask_correct_corr = self.sample_valid_image_pair()
        # corres_map_self_to_other_ (H, W, 2)
        # conf_map_self_to_other_ (H, W, 1)
        # variance_self_to_other_ (H, W, 1)
        # mask_correct_corr (H, W, 1)

        if iteration < self.opt.precrop_iters:
            # in case we only want to compute renderings in the image center
            # exclude outside pixels from the valid mask
            H, W = data_dict.image.shape[-2:]
            dH = int(H//2 * self.opt.precrop_frac)
            dW = int(W//2 * self.opt.precrop_frac)
            # we only want to sample in the center of the image
            mask_center = torch.zeros_like(mask_correct_corr)  # (H, W, 1)
            mask_center[H//2 - dH:H//2 + dH - 1, W//2 - dW:W//2 + dW - 1] = 1
            mask_correct_corr = mask_correct_corr & mask_center

        return self.compute_loss_at_given_img_indexes(opt, data_dict, id_self, id_matching_view, corres_map_self_to_other_, 
                                                      conf_map_self_to_other_, variance_self_to_other_, mask_correct_corr, 
                                                      loss_dict, stats_dict, plotting_dict, plot)

    def compute_loss_at_given_img_indexes(self, opt: Dict[str, Any], data_dict: Dict[str, Any], id_self: int, 
                                          id_matching_view: int, corres_map_self_to_other_: torch.Tensor, 
                                          conf_map_self_to_other_: torch.Tensor, variance_self_to_other_: torch.Tensor, 
                                          mask_correct_corr: torch.Tensor, loss_dict: Dict[str, Any], 
                                          stats_dict: Dict[str, Any], plotting_dict: Dict[str, torch.Tensor], 
                                          plot: bool=False, skip_verif: bool=True
                                          )-> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - pose: gt w2c poses (B, 3, 4)
                            - pose_w2c: current estimates of w2c poses (B, 3, 4). When the camera poses
                            are fixed to gt, pose=pose_w2c. Otherwise, pose_w2c is being optimized. 
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            id_self (int): index of first image in pair
            id_matching_view (int): index of second image in pair
            corres_map_self_to_other_ (torch.Tensor): (H, W, 2)
            conf_map_self_to_other_ (torch.Tensor): (H, W, 1)
            variance_self_to_other_ (torch.Tensor): (H, W, 1) or None
            mask_correct_corr (torch.Tensor): (H, W, 1), valid correspondences 
                                              (which can be used for the loss)
            loss_dict (dict): dictionary with loss values 
            stats_dict (dict)
            plotting_dict (dict)
            plot (bool)
        """
        # data_dict also corresponds to the whole scene here 
        # here adds the size of the image, because used a lot later
        B, _, H, W = data_dict.image.shape
        images = data_dict.image.permute(0, 2, 3, 1)  # (B, 3, H, W) then (B, H, W, 3)

        # have the poses in output_dict (at current iteration)
        poses_w2c = data_dict.poses_w2c  # current estimate of the camera poses
        intrs = data_dict.intr

        pose_w2c_self = torch.eye(4).to(poses_w2c.device)
        pose_w2c_self[:3, :4] = poses_w2c[id_self] # the pose itself is just (3, 4)
        pose_w2c_other = torch.eye(4).to(poses_w2c.device)
        pose_w2c_other[:3, :4] = poses_w2c[id_matching_view]  # (3, 4)

        intr_self, intr_other = intrs[id_self], intrs[id_matching_view]  # (3, 3)

        # for correspondence loss 
        corres_map_self_to_other = corres_map_self_to_other_.detach() # [h,w,2], pixel to pixel matching
        conf_map_self_to_other = conf_map_self_to_other_.detach() # [h,w,1]
        mask_correct_corr = mask_correct_corr.detach().squeeze(-1)  # (H, W)
        corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()  # (H, W, 2)
        corres_map_self_to_other_rounded_flat = \
            corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
        # (H, W), every scalar is an index


        # (H, W, 1) and then (H, W)  bool.Tensor
        with torch.no_grad():
            # if gt corrspondences are available, have an idea of the quality of the predicted
            # correspondences
            if (not skip_verif or self.opt.use_gt_correspondences) \
                and ('depth_gt' in data_dict.keys() or self.opt.use_gt_correspondences):
                corres_map_and_mask_self_to_other_gt = self.gt_corres_map_and_mask_all_to_all[id_self, id_matching_view]  # (3, H, W)
                corres_map_self_to_other_gt = corres_map_and_mask_self_to_other_gt[:2].permute(1, 2, 0) # (H, W, 2)
                mask_correct_corr_ = corres_map_and_mask_self_to_other_gt[-1].bool()  # (H, W)
                
                error = torch.norm(corres_map_self_to_other.float()-corres_map_self_to_other_gt.float(), dim=-1, keepdim=True)
                error_all = error[mask_correct_corr_]
                stats_dict['epe_all'] = error_all.mean() if len(error_all) > 0 else torch.as_tensor(0.)
                stats_dict['pck_1_all'] = error_all.le(1.).float().mean() if len(error_all) > 0 else torch.as_tensor(0.)
                stats_dict['pck_3_all'] = error_all.le(3.).float().mean() if len(error_all) > 0 else torch.as_tensor(0.)

                error_conf = error[mask_correct_corr & mask_correct_corr_]
                stats_dict['epe_in_conf'] = error_conf.mean() if len(error_conf) > 0 else torch.as_tensor(0.)
                stats_dict['pck_1_in_conf'] = error_conf.le(1.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)
                stats_dict['pck_3_in_conf'] = error_conf.le(3.).float().mean() if len(error_conf) > 0 else torch.as_tensor(0.)

                if  self.opt.use_gt_correspondences:
                    # debugging with gt correspondences
                    # DEBUGGING
                    # replace the correspondences that are predicted
                    corres_map_self_to_other = corres_map_self_to_other_gt.clone()
                    corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).to(self.device).long()
                    corres_map_self_to_other_rounded_flat = corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
                    if self.opt.use_dummy_all_one_confidence:
                        mask_correct_corr = mask_correct_corr_
                    else:
                        mask_correct_corr = mask_correct_corr & mask_correct_corr_
            
        if mask_correct_corr.sum() < self.opt.min_nbr_matches: # [False,True,...] < 500
            # print('Matching {} to {}: did not find enough correspondences'.format(id_self, id_matching_view))
            # didnt find any valid correspondences
            return loss_dict, stats_dict, plotting_dict
        stats_dict['perc_valid_corr_mask'] = mask_correct_corr.sum() / (mask_correct_corr.nelement() + 1e-6)  # 17%
        # * according to image pair, define pose, corrs_map, conf_map
        return self.compute_loss_on_image_pair(data_dict, images, poses_w2c, intrs, id_self, id_matching_view, 
                                               corres_map_self_to_other, corres_map_self_to_other_rounded_flat, 
                                               conf_map_self_to_other, mask_correct_corr, pose_w2c_self, pose_w2c_other, intr_self, 
                                               intr_other, loss_dict, stats_dict, plotting_dict)


    
        
        
        
    # * track optim
    @torch.no_grad()
    def refine_keypoint(self,combi_list,train_data):
        import os,sys
        from PIL import Image
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent / '../../../third_party/Hierarchical-Localization'))
        from hloc import pairs_from_exhaustive
        from validation.utils import matches_from_flow
        from hloc.utils.parsers import names_to_pair # older version using _ as separator
        sys.path.append(str(Path(__file__).parent / '../../utils/colmap_initialization'))
        from pdcnet_for_hloc import remove_duplicates
        import pickle
        assert len(train_data.all.rgb_path)>0
        
        def move_pixel_maps_to_device():
            for idx in range(len(self.pixels_map)):
                self.pixels_map[idx][0]=self.pixels_map[idx][0].to(self.device)
                self.pixels_map[idx][1]=self.pixels_map[idx][1].to(self.device)
                
                
        if self.opt.track.precompute_ka_for_corrs_loss:
            # get dataset name and scan name 
            precomputed_path=os.path.join(self.opt.env.precomputed,self.opt.dataset,self.opt.scene,'pixelsmap.pkl')
            if os.path.isfile(precomputed_path):
                print(f"Using precomputed pixels map from {precomputed_path}")
                with open(precomputed_path, 'rb') as file:
                    self.pixels_map = pickle.load(file)
                move_pixel_maps_to_device()
                return
            else:
                print(f"Warning: precomputed pixels map from {precomputed_path} not found. Re-calculating...")
        
        
        
        
        
        
        from pixsfm.refine_hloc import PixSfM
        abs_rgb_path=[os.path.join(train_data.path_image, x) for x in train_data.all.rgb_path]
        # check the actual image size matches with the one we get in, essential for kp match
        example_path=abs_rgb_path[0]
        assert os.path.isfile(example_path)
        _test_img=Image.open(example_path)
        H, W = train_data.all.image.shape[-2:]
        _W,_H=_test_img.size
        # * assert H==_H and W==_W, f"Source file resolution {_H} {_W} is not equal to loaded resolution {H} {W}."
        if not (H==_H and W==_W):
            print(f"Warning: Source file resolution {_H} {_W} is not equal to loaded resolution {H} {W}.")
            # current we suppose the scale for h,w is the same, but it should be easily changed
            assert _H/H==_W/W
            scaling_kp = _H/H
        else:
            scaling_kp=1.0
        
        # angle pair is not supported because of matching loop 
        assert self.opt.matching_pair_generation in ['all','all_to_all']
        num_keypoints=train_data.all.image.shape[-1]*train_data.all.image.shape[-2]
        
        combi_list_t = (combi_list.cpu().numpy().T).tolist()  
        self.pixels_map=[[None, None] for _ in range(len(combi_list_t))]
        
        
        b2a = [[i]+t for i,t in enumerate(combi_list_t) if t[0] > t[1]]
        a2b = [[i]+t for i,t in enumerate(combi_list_t) if t[0] < t[1]]
        image_names=[os.path.basename(path) for path in abs_rgb_path]
    
        feature_path=Path(self.opt.env.workspace_dir,'keypoints.h5')
        feature_file = h5py.File(str(feature_path), 'w')
        # kp_dict={}
        for image_name in image_names:
            temp_keypoints=get_grid_keypoints({'size_original': (H,W)}, edict({'scaling_kp':1.0}))
            temp_keypoints *= scaling_kp
            # nd_array, (w,h), note this is keypoint format
            # kp_dict['keypoints']=temp_keypoints
            grp = feature_file.create_group(image_name) # (0,0)...(377,0)...(377,504), (idx_w,idx_h)
            grp.create_dataset('keypoints', data=temp_keypoints) # in shape [N,2], N=h*w
        feature_file.close()
        
        # * note : we must use original scale to select keypoints
        size_of_keypoints_s=(H,W)
        size_of_keypoints_t=(H,W)
        
        
        # split the filtered_flow_pairs into two topology direction
        for idx,match_loop in enumerate([a2b,b2a]):
            if len(match_loop)==0:
                assert idx==1 # default pair
                continue
            # matches_dict={}
            match_file=Path(os.path.join(self.opt.env.workspace_dir, f'match_{idx}.h5'))
            match_file_writer = h5py.File(match_file, 'w')
            pairs_file=Path(os.path.join(self.opt.env.workspace_dir, f'pairs_{idx}.txt'))
            feature_output_path=Path(self.opt.env.workspace_dir,f'keypoints_{idx}.h5')
            
            # generate pair file, third_party/Hierarchical-Localization/hloc/pairs_from_exhaustive.py
            pairs_from_exhaustive.main(output=pairs_file, image_list=image_names if idx==0 else image_names[::-1])
            """
            # get match dict
            # self.corres_maps, self.conf_maps, self.mask_valid_corr, self.filtered_flow_pairs
            # example retrieval in loss: 
                corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).to(self.device).long()
                corres_map_self_to_other_rounded_flat = corres_map_self_to_other_rounded[:, :, 1] * W + 
                        corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
                 self.grid = torch.stack((xx, yy), dim=-1).to(self.device).float()  # ( H, W, 2)
                pixels_in_other = corres_map_self_to_other[mask_correct_corr] # [N_ray, 2], absolute pixel locations, float
                pixels_in_self = self.grid[mask_correct_corr]  # [N_ray, 2], absolute pixel locations of correct
            """
            
            for pair in match_loop: # pair: [pair_idx, target_img, source_img]
                
                
                # matches_dict[names_to_pair(image_names[pair[2]], image_names[pair[1]])] = matches
                id_in_flow_tensor, id_self, id_matching_view = pair[0], pair[1], pair[2]
                corres_map_self_to_other_ = self.corres_maps[id_in_flow_tensor].permute(1, 2, 0)[:, :, :2]  # (H, W, 2)
                conf_map_self_to_other_ = self.conf_maps[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
                variance_self_to_other_ = None
                mask_correct_corr = self.mask_valid_corr[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
                
                # for correspondence 
                corres_map_self_to_other = corres_map_self_to_other_.detach() # [h,w,2], pixel to pixel matching
                conf_map_self_to_other = conf_map_self_to_other_.detach() # [h,w,1]
                mask_correct_corr = mask_correct_corr.detach().squeeze(-1)  # (H, W)
                corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()  # (H, W, 2)
                corres_map_self_to_other_rounded_flat = \
                    corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
                
                # grid: (h,w,2), element: (idx_w,idx_h)
                pixels_in_self = self.grid[mask_correct_corr].cpu().numpy()  # [N_ray, 2], absolute pixel locations of correct
                pixels_in_other = corres_map_self_to_other[mask_correct_corr].cpu().numpy() # [N_ray, 2], absolute pixel locations, float
                conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1] 
                
                pixels_in_self_rounded=np.int32(np.round(pixels_in_self))
                pixels_in_other_rounded = np.int32(np.round(pixels_in_other))
                conf_values = conf_values.cpu().numpy()
                
                name_of_pair=names_to_pair(image_names[pair[1]], image_names[pair[2]])
                # XY matches 
                idx_A = (pixels_in_self_rounded[:, 1] * W + pixels_in_self_rounded[:, 0]).reshape(-1) # flatten img0 pixel index
                idx_B = (pixels_in_other_rounded[:, 1] * W + pixels_in_other_rounded[:, 0]).reshape(-1) # flatten img1 pixel index
                
                
                grp = match_file_writer.create_group(name_of_pair)
                # save the matches in original hloc format
                matches0_hloc=np.full(num_keypoints,-1)
                scores=np.full(num_keypoints,0,dtype=np.float64)
                
                # Updating matches0 array
                for k, ka in enumerate(idx_A): # k-th flow point in valid match
                    matches0_hloc[ka] = idx_B[k] # ka-th keypoint in image
                    scores[ka]=conf_values[k]
                grp.create_dataset('matches0', data=matches0_hloc) # matches: [1783], int, array([-1, 46, -1, ..., -1, -1, -1]
                grp.create_dataset('matching_scores0', data=scores) # # [1783], range 0-1
                    
                # finished for every match pair
            match_file_writer.close()
            
            
            
            # perform the track optimization
            # reload the matches into self. 
            # scores ? no need to change, only thing to do next: keypoints location may changed....
            """
            # from pixsfm.refine_hloc import PixSfM
            # refiner = PixSfM()
            # keypoints, _, _ = refiner.refine_keypoints(
            #     path_to_output_keypoints.h5,
            #     path_to_input_keypoints.h5,
            #     path_to_list_of_image_pairs,
            #     path_to_matches.h5,
            #     path_to_image_dir,
            # )
            """
            
            refiner = PixSfM()
            kps_new,ka_data,feature_manager=refiner.refine_keypoints(output_path=feature_output_path,
                                             features_path=feature_path,
                                             image_dir=Path(example_path).parent,
                                             pairs_path=pairs_file,
                                             matches_path=match_file
                                             )
            for k in kps_new.keys():
                kps_new[k]/=scaling_kp
            
            for pair in match_loop: 
                # matches_dict[names_to_pair(image_names[pair[2]], image_names[pair[1]])] = matches
                id_in_flow_tensor, id_self, id_matching_view = pair[0], pair[1], pair[2]
                corres_map_self_to_other_ = self.corres_maps[id_in_flow_tensor].permute(1, 2, 0)[:, :, :2]  # (H, W, 2)
                conf_map_self_to_other_ = self.conf_maps[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
                variance_self_to_other_ = None
                mask_correct_corr = self.mask_valid_corr[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
                
                # for correspondence 
                corres_map_self_to_other = corres_map_self_to_other_.detach() # [h,w,2], pixel to pixel matching
                conf_map_self_to_other = conf_map_self_to_other_.detach() # [h,w,1]
                mask_correct_corr = mask_correct_corr.detach().squeeze(-1)  # (H, W)
                corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()  # (H, W, 2)
                corres_map_self_to_other_rounded_flat = \
                    corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
                
                
                pixels_in_self = self.grid[mask_correct_corr].cpu().numpy()  # [N_ray, 2], absolute pixel locations of correct
                pixels_in_other = corres_map_self_to_other[mask_correct_corr].cpu().numpy() # [N_ray, 2], absolute pixel locations, float
                conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1] 
                # grid: (h,w,2), element: (idx_w,idx_h)
                pixels_in_self_rounded=np.int32(np.round(pixels_in_self))
                pixels_in_other_rounded = np.int32(np.round(pixels_in_other))
                conf_values = conf_values.cpu().numpy()
                
                name_of_pair=names_to_pair(image_names[pair[1]], image_names[pair[2]])
                # XY matches 
                idx_A = (pixels_in_self_rounded[:, 1] * W + pixels_in_self_rounded[:, 0]).reshape(-1) # flatten img0 pixel index
                idx_B = (pixels_in_other_rounded[:, 1] * W + pixels_in_other_rounded[:, 0]).reshape(-1) # flatten img1 pixel index
                
                # * reload the optimized keypoints back to memory
                source_kps,target_kps=kps_new[image_names[pair[1]]],kps_new[image_names[pair[2]]]
                pixels_in_self_opt,pixels_in_other_opt=source_kps[idx_A],target_kps[idx_B]
                pixels_in_self_opt_,pixels_in_other_opt_=torch.from_numpy(pixels_in_self_opt),torch.from_numpy(pixels_in_other_opt)
                #  why here keypoints format became (idx_h, idx_w)
                self.pixels_map[id_in_flow_tensor][0]=pixels_in_self_opt_.float()
                self.pixels_map[id_in_flow_tensor][1]=pixels_in_other_opt_.float()
        
        if self.opt.track.precompute_ka_for_corrs_loss:
            Path(os.path.dirname(precomputed_path)).mkdir(exist_ok=True,parents=True)
            with open(precomputed_path, 'wb') as file:
                pickle.dump(self.pixels_map, file)
            print(f"Precomputed pixels map from {precomputed_path} saved.")
            
        move_pixel_maps_to_device()
        return
