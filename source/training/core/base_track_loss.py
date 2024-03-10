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
from source.training.core.track_utils import (generate_pair_list, 
                                                       image_pair_candidates_with_angular_distance, 
                                                       TrackUtils, get_mask_valid_from_conf_map)
from source.utils.config_utils import override_options
from source.utils.colmap_initialization.pdcnet_for_hloc import get_grid_keypoints
import h5py
from source.utils.camera import pose_inverse_4x4
from source.utils.geometry.batched_geometry_utils import batch_project_to_other_img
 
class TrackBasedLoss(BaseLoss, TrackUtils):
    """Track Loss. Main signal for the joint pose-NeRF training. """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, flow_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(device=device)
        default_cfg = edict({'matching_pair_generation': 'all', 
                             'min_nbr_matches': 500, 
                             'pairing_angle_threshold': 30, # degree, in case 'angle' pair selection chosen
                             'filter_corr_w_cc': False, 
                             'min_conf_valid_corr': 0.95, 
                             'min_conf_cc_valid_corr': 1./(1. + 1.5), 
                              # for deeper loss calculation, similar to corres_loss
                             'diff_loss_type': 'huber', 
                             'compute_photo_on_matches': False, 
                             'renderrepro_do_pixel_reprojection_check': False, 
                             'renderrepro_do_depth_reprojection_check': False, 
                             'renderrepro_pixel_reprojection_thresh': 10., 
                             'renderrepro_depth_reprojection_thresh': 0.1, 
                             'use_gt_depth': False,  # debugging
                             'use_gt_correspondences': False,  # debugging
                             'use_dummy_all_one_confidence': False # debugging
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

        self.compute_correspondences_and_track(train_data) # * note : the 2d-2d matching is computed from here
        # flow will either be computed on the fly (if weights are finetuned or not)

    @torch.no_grad()
    def compute_correspondences_and_track(self, train_data: Dict[str, Any]):
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

        self.corres_maps = corres_maps  # (combi_list.shape[1], 3, H, W), already input rgb size
        self.conf_maps = conf_maps
        self.mask_valid_corr = mask_valid_corr # * note we save the mask here and then use it for sampling
        # should be list of the matching index for each of the image. 
        # first row/element corresponds to the target image, second is the source image
        flow_pairs = (combi_list.cpu().numpy().T).tolist()  
        self.flow_pairs = flow_pairs
        # list of pairs, the target is the first element, source is second
        assert corres_maps.shape[0] == len(flow_pairs)
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
        if self.opt.track.enable_ka_for_track_loss:
            self.refine_keypoint(combi_list, train_data)
            
            
        
        
        return 


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
        
        loss_dict, stats_dict, plotting_dict = self.compute_loss_track_graph\
            (opt, data_dict, output_dict, iteration, mode, plot)
        
        if self.opt.gradually_decrease_corres_weight: #  default false
            # gamma = 0.1**(max(iteration - self.opt.start_iter_photometric, 0)/self.opt.max_iter)
            # reduce the corres weight by 2 every x iterations 
            iter_start_decrease_corres_weight = self.opt.ratio_start_decrease_corres_weight * self.opt.max_iter \
                if self.opt.ratio_start_decrease_corres_weight is not None \
                    else self.opt.iter_start_decrease_corres_weight
            if iteration < iter_start_decrease_corres_weight:
                gamma = 1. 
            else:
                gamma = 2 ** ((iteration-iter_start_decrease_corres_weight) // self.opt.corres_weight_reduct_at_x_iter)
            loss_dict['tracks'] = loss_dict['tracks'] / gamma
        # cfg.iter_start_decrease_corres_weight = 0
        # cfg.corres_weight_reduct_at_x_iter = 10000
        if not self.track_graph[0].enough_track:
            gamma=iteration/self.opt.max_iter
            loss_dict['tracks'] = loss_dict['tracks'] / (gamma*100.)
        return loss_dict, stats_dict, plotting_dict

    
    def compute_loss_track_graph(self, opt: Dict[str, Any], data_dict: Dict[str, Any], 
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
        # todo : change the vis behavior
        stats_dict, plotting_dict, loss_dict = {}, {}, {'tracks': torch.tensor(0., requires_grad=True).to(self.device)}
                                                       # 'render_tracks': torch.tensor(0., requires_grad=True).to(self.device)}
        
        if mode != 'train':
            # only during training
            return loss_dict, stats_dict, plotting_dict

        if iteration < self.opt.start_iter.corres: # 0
            # if the correspondence loss is only added after x iterations
            return loss_dict, stats_dict, plotting_dict

        if len(self.filtered_flow_pairs) == 0:
            return loss_dict, stats_dict, plotting_dict
        # * track sampling happens here, sample is counted by tracks
       
        for track_graph in self.track_graph:
            sampled_tracks=track_graph.sample_tracks()
            
            to_render_pixels_dict,node_record_dict=track_graph.tracks_to_image_pixels(sampled_tracks)
        
            temp_loss_dict,temp_stats_dict,temp_plotting_dict=self.compute_loss_at_given_tracks(opt,data_dict,
                track_graph,sampled_tracks,to_render_pixels_dict,node_record_dict,iteration)
           
            loss_dict['tracks']+=temp_loss_dict['tracks']

        return loss_dict,temp_stats_dict,temp_plotting_dict 
        # * note, since we have two track graph, optionally return one to above
        

    def compute_loss_at_given_tracks(self, opt: Dict[str, Any], data_dict: Dict[str, Any], 
                                          track_graph,sampled_tracks,to_render_pixels_dict,node_record_dict,iteration,
                                          plot: bool=False, skip_verif: bool=True
                                          )-> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
        
            loss_dict (dict): dictionary with loss values 
            stats_dict (dict)
            plotting_dict (dict)
            plot (bool)
        """
        stats_dict, plotting_dict, loss_dict = {}, {}, {'tracks': torch.tensor(0., requires_grad=True).to(self.device), 
                                                        'render_tracks': torch.tensor(0., requires_grad=True).to(self.device)}
        
        # data_dict also corresponds to the whole scene here 
        # here adds the size of the image, because used a lot later
        B, _, H, W = data_dict.image.shape
        images = data_dict.image.permute(0, 2, 3, 1)  # (B, 3, H, W) then (B, H, W, 3)

        # have the poses in output_dict (at current iteration)
        poses_w2c = data_dict.poses_w2c  # current estimate of the camera poses
        intrs = data_dict.intr

        # * changing the self/other camera pose to involved images
        to_render_img_ids=sorted(list(to_render_pixels_dict.keys()))
        to_render_intrs={x:intrs[x] for x in to_render_img_ids}
        to_render_pose_w2c={img_id:torch.eye(4).to(poses_w2c.device) for img_id in to_render_img_ids}
        for img_id in to_render_img_ids:
            to_render_pose_w2c[img_id][:3,:4]=poses_w2c[img_id]

        to_render_pixels_rounded_dict={k:torch.round(v).long() for k,v in to_render_pixels_dict.items()}
        to_render_ray_int_dict={k:v[:,1]*W+v[:,0] for k,v in to_render_pixels_rounded_dict.items()}
    

  
        selected_total_pixels=np.sum([v.shape[0] for v in to_render_pixels_dict.values()]) # 1971
        if selected_total_pixels.sum() < self.opt.min_nbr_matches: # [False,True,...] < 500
            print('Track problem for images {}: did not find enough correspondences under min {} rays'.format(to_render_img_ids,self.opt.min_nbr_matches))
            # should not happen as we are using random.choice which enables duplicate, but keep the logic for now
            return loss_dict, stats_dict, plotting_dict
        
        # the actual render and reproject code
        iteration = data_dict['iter']
        
        ret_dict={}
        for img_id,pixels in to_render_pixels_dict.items():
            ret_dict[img_id] = self.net.render_image_at_specific_pose_and_rays(self.opt, data_dict, to_render_pose_w2c[img_id][:3], 
                                                                   to_render_intrs[img_id], H, W, 
                                                                   pixels=pixels, mode='train', iter=iteration)
            
        reproj_dict=self.project_all_imgs(ret_dict,to_render_pixels_dict,to_render_pose_w2c,to_render_intrs)
        # how about refractor it as node_idx 
        
        
        # it returns 'origins', 'viewdirs', 'rgb_samples', 'density_samples', 't', 'rgb', 'rgb_var', 'depth', 
        # 'depth_var', 'opacity', 'weights', 'all_cumulated', 'ray_idx'
        rendered_dict=track_graph.get_pixels_for_sampled_tracks(
            sampled_tracks,to_render_ray_int_dict,node_record_dict,ret_dict,images)
        
        # (track_num,track_len)
        all_track_colors,all_gt_colors,all_root_idx=rendered_dict['all_track_colors'],rendered_dict['all_gt_colors'],rendered_dict['all_root_idx']
        
        if self.opt.compute_photo_on_matches: # * default false, no plan to add it now
            for tidx,track in enumerate(sampled_tracks):
                # we have two options, make it close to root pixels, or make it close to respected pixels on related images
                if self.opt.photo_loss=='root':
                    gt_root_colors=all_gt_colors[tidx][all_root_idx[tidx]] # (1,3)
                    color_this_track=all_track_colors[tidx] # (track.len,3)
                    loss_photo=self.MSE_loss_many_to_one(color_this_track,gt_root_colors)
                    if 'all_track_fine_colors' in rendered_dict.keys():
                        color_fine_this_track=rendered_dict['all_track_fine_colors'][tidx] # (track.len,3)
                        loss_photo+=self.MSE_loss_many_to_one(color_fine_this_track,gt_root_colors)
                elif self.opt.photo_loss=='multi':
                    gt_track_colors=all_gt_colors[tidx]
                    color_this_track=all_track_colors[tidx] # (track.len,3)
                    loss_photo=self.MSE_loss_many_to_many(color_this_track,gt_track_colors)
                    if 'all_track_fine_colors' in rendered_dict.keys():
                        color_fine_this_track=rendered_dict['all_track_fine_colors'][tidx] # (track.len,3)
                        loss_photo+=self.MSE_loss_many_to_many(color_fine_this_track,gt_track_colors)
            loss_dict['render_tracks'] += loss_photo
            # weighted by track length but should scale to pixel wise? maybe no need, like /track.len ??
        
        
        # stats_dict['depth_in_track_loss'] = rendered_dict['all_track_depths'].detach().mean()
        
        # reprojection loss 1: project root to others
        # reprojection loss 2: project others to root
        total_track_loss, total_loss_weight=[], []
        for tidx,track in enumerate(sampled_tracks):
            track_loss= torch.tensor(0., requires_grad=True).to(self.device)
            root_node=track.nodes[all_root_idx[tidx]] # root node idx in the track -> get root node
            
            root_node_img_id,root_node_idx_in_todo=node_record_dict[root_node.node_idx]
            root_node_pixel_coord=to_render_pixels_dict[root_node_img_id][root_node_idx_in_todo]
            
            # the loss is accumulated pixel by pixel, one by one, not track wise
            for node in track.nodes:
                if self.opt.track.mode=='root':
                    if node.is_root:
                        continue
                    node_img_id,node_idx_in_todo=node_record_dict[node.node_idx]
                    node_pixel_coord=to_render_pixels_dict[node_img_id][node_idx_in_todo]
                    
                    # proj=f(source,depth,T), loss=proj-target, target=matcher(source), gradients flow into T and depth (NeRF)
                    root_node_pixel_coord_in_other=reproj_dict[root_node_img_id][node_img_id]['pixels_proj2other'][root_node_idx_in_todo]
                    other_node_pixel_coord_in_root=reproj_dict[node_img_id][root_node_img_id]['pixels_proj2other'][node_idx_in_todo]
                    
                    loss_track = self.compute_diff_loss(loss_type=self.opt.diff_loss_type, 
                                                        diff=node_pixel_coord - root_node_pixel_coord_in_other, 
                                                        weights=None, mask=torch.ones_like(root_node_pixel_coord), dim=-1)
                    
                    loss_track_ = self.compute_diff_loss(loss_type=self.opt.diff_loss_type,
                                                        diff=root_node_pixel_coord - other_node_pixel_coord_in_root, 
                                                        weights=None, mask=torch.ones_like(node_pixel_coord), dim=-1)
                    
                    if 'all_track_fine_depths' in rendered_dict.keys():
                        fine_root_node_pixel_coord_in_other=reproj_dict[root_node_img_id][node_img_id]['fine_pixels_proj2other'][root_node_idx_in_todo]
                        fine_other_node_pixel_coord_in_root=reproj_dict[node_img_id][root_node_img_id]['fine_pixels_proj2other'][node_idx_in_todo]
                        loss_track += self.compute_diff_loss(loss_type=self.opt.diff_loss_type, 
                                                        diff=node_pixel_coord - fine_root_node_pixel_coord_in_other, 
                                                        weights=None, mask=torch.ones_like(root_node_pixel_coord), dim=-1)
                    
                        loss_track_ += self.compute_diff_loss(loss_type=self.opt.diff_loss_type,
                                                        diff=root_node_pixel_coord - fine_other_node_pixel_coord_in_root, 
                                                        weights=None, mask=torch.ones_like(node_pixel_coord), dim=-1)
                        
                        track_loss += (loss_track + loss_track_ ) / 4.
                    else:
                        track_loss += (loss_track + loss_track_ ) / 2.
                elif self.opt.track.mode=='mv':
                    node_img_id,node_idx_in_todo=node_record_dict[node.node_idx]
                    node_pixel_coord=to_render_pixels_dict[node_img_id][node_idx_in_todo]
                    # project other to root
                    for root_node in track.nodes:
                        root_node_img_id,root_node_idx_in_todo=node_record_dict[root_node.node_idx]
                        root_node_pixel_coord=to_render_pixels_dict[root_node_img_id][root_node_idx_in_todo]
                        if root_node_idx_in_todo==node_idx_in_todo:
                            # same node, skip
                            continue
                        other_node_pixel_coord_in_root=reproj_dict[node_img_id][root_node_img_id]['pixels_proj2other'][node_idx_in_todo]
                        loss_track_ = self.compute_diff_loss(loss_type=self.opt.diff_loss_type,
                                                        diff=root_node_pixel_coord - other_node_pixel_coord_in_root, 
                                                        weights=None, mask=torch.ones_like(node_pixel_coord), dim=-1)
                        if 'all_track_fine_depths' in rendered_dict.keys():
                            fine_other_node_pixel_coord_in_root=reproj_dict[node_img_id][root_node_img_id]['fine_pixels_proj2other'][node_idx_in_todo]
                            loss_track_ += self.compute_diff_loss(loss_type=self.opt.diff_loss_type,
                                                        diff=root_node_pixel_coord - fine_other_node_pixel_coord_in_root, 
                                                        weights=None, mask=torch.ones_like(node_pixel_coord), dim=-1)
                            track_loss += loss_track_  / 4.
                        else:
                            track_loss += loss_track_  / 2.
                else:
                    raise NotImplementedError()
                        
                        
                    
               
                """
                1. Traverse all node, find it's projection on every other except it self, loss+=, just project it to others,
                and then record how many time we have called compute_diff_loss, /num_times, for the weight ? sum of node score !
                """
            if self.opt.model=='nerf_gt_poses' or (self.opt.track.mode=='mv' and iteration>self.opt.max_iter*self.opt.ratio_end_joint_nerf_pose_refinement):
                # enable length normalization during the training, but we 
                track_loss = track_loss / float(track.len) 
            total_track_loss.append(track_loss)
            total_loss_weight.append(root_node.score)
        
        total_track_loss= torch.stack(total_track_loss)
        if self.opt.track.mode=='mv': # ignore root node weight, since match score always > 0.95, treat all matches equally
            total_loss_weight=torch.ones_like(total_track_loss).to(total_track_loss.device).to(total_track_loss.dtype)
        else: # decide by the root node
            total_loss_weight= torch.tensor(total_loss_weight).to(total_track_loss.device).to(total_track_loss.dtype)
        weighted_sum = torch.sum(total_track_loss * total_loss_weight)
        total_weight = torch.sum(total_loss_weight)
        loss_dict['tracks']= weighted_sum / total_weight
        return loss_dict, stats_dict, plotting_dict
    

    
    def MSE_loss(self, pred: torch.Tensor, label: torch.Tensor):
        loss = (pred.contiguous()-label)**2
        return loss.sum() / (loss.nelement() + 1e-6)
    
    def MSE_loss_many_to_one(self, pred: torch.Tensor, label: torch.Tensor):
        # pred shape: (N,3), label shape: (1,3)
        # Expand label to match the shape of pred
        label_expanded = label.expand_as(pred)
        return self.MSE_loss(pred, label_expanded)
    
    def MSE_loss_many_to_many(self, pred: torch.Tensor, label: torch.Tensor):
        # pred shape: (N,3), label shape: (N,3)
        return self.MSE_loss(pred, label)
    
    def project_all_imgs(self,ret_dict,to_render_pixels_dict,to_render_pose_w2c,intrs):
        reproj_dict={}
        for self_img_id in ret_dict.keys():
            reproj_dict[self_img_id]={}
            for other_img_id in ret_dict.keys():
                if self_img_id==other_img_id: continue
                ret_self=ret_dict[self_img_id]
                depth_rendered_self = ret_self.depth.squeeze(0).squeeze(-1)
                # * note
                # depth_rendered_other = ret_other.depth.squeeze(0).squeeze(-1)
                # opt.renderrepro_do_depth_reprojection_check is default false in our case
                pose_w2c_self=to_render_pose_w2c[self_img_id]
                pose_w2c_other=to_render_pose_w2c[other_img_id]
                T_self2other = pose_w2c_other @ pose_inverse_4x4(pose_w2c_self)
                intr_self, intr_other = intrs[self_img_id], intrs[other_img_id]  # (3, 3)
                
                pixels_in_self=to_render_pixels_dict[self_img_id]
                # pixels_in_other=to_render_pixels_dict[other_img_id]
        
                pts_self_repr_in_other, depth_self_repr_in_other = batch_project_to_other_img(
                    pixels_in_self, di=depth_rendered_self, 
                    Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)
                
                reproj_dict[self_img_id][other_img_id]={}
                reproj_dict[self_img_id][other_img_id]['pixels_proj2other']=pts_self_repr_in_other
                # reproj_dict[self_img_id][other_img_id]['depths_proj2other']=depth_self_repr_in_other,
                # see above note, we don't do depth reprojection check
                # we just do project here, without considering correspondence at this moment
                # they will be treated as label to supervise other
                
                if 'depth_fine' in ret_self.keys():
                    fine_depth_rendered_self = ret_self.depth_fine.squeeze(0).squeeze(-1)
                    fine_pts_self_repr_in_other, fine_depth_self_repr_in_other = batch_project_to_other_img(
                    pixels_in_self, di=fine_depth_rendered_self, 
                    Ki=intr_self, Kj=intr_other, T_itoj=T_self2other, return_depth=True)
                    reproj_dict[self_img_id][other_img_id]['fine_pixels_proj2other']=fine_pts_self_repr_in_other
                    
        return reproj_dict
                
        
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
        from source.training.core.track_graph import Node,Track,TrackGraph,export_trackgraph
        import pickle
        assert len(train_data.all.rgb_path)>0
        
        
        
        if self.opt.track.precompute_ka_for_track_loss:
            # get dataset name and scan name 
            precomputed_path=os.path.join(self.opt.env.precomputed,self.opt.dataset,self.opt.scene,f'trackgraph_{self.opt.train_sub}.pkl')
            if os.path.isfile(precomputed_path):
                if hasattr(self.opt.track,'overwrite') is False or self.opt.track.overwrite is False:
                    print(f"Using precomputed track graph from {precomputed_path}")
                    with open(precomputed_path, 'rb') as file:
                        self.track_graph = pickle.load(file)
                        # * note : try with one graph for consistency
                        if hasattr(self.opt.track,'graph') and self.opt.track.graph=='single':
                            self.track_graph = self.track_graph[:1]
                        if self.track_graph[0].enough_track is False:
                            # todo
                            print("Warning: No enough tracks detected in this scene. Lower the confidence of track loss.")
                    return
                elif self.opt.track.overwrite is True:
                    print(f"Warning: Overwrite precomputed track graph from {precomputed_path} . Re-calculating...")
            else:
                print(f"Warning: precomputed track graph from {precomputed_path} not found. Re-calculating...")
        
        
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
            print(f"Feature extraction working on a resolution of W={W} and H={H}.")
            scaling_kp=1.0
        # angle pair is not supported because of matching loop 
        assert self.opt.matching_pair_generation in ['all','all_to_all']
        num_keypoints=train_data.all.image.shape[-1]*train_data.all.image.shape[-2]
        
        combi_list_t = (combi_list.cpu().numpy().T).tolist()  
        # self.pixels_map=[[None, None] for _ in range(len(combi_list_t))]
        
        
        b2a = [[i]+t for i,t in enumerate(combi_list_t) if t[0] > t[1]]
        a2b = [[i]+t for i,t in enumerate(combi_list_t) if t[0] < t[1]]
        image_names=[os.path.basename(path) for path in abs_rgb_path]
        # we need this mapping to match with track info 
        self.image_name_to_id={x:idx for idx,x in enumerate(image_names)}
    
        feature_path=Path(self.opt.env.workspace_dir,'keypoints_tg.h5')
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
        self.track_graph=[]
        for idx,match_loop in enumerate([a2b,b2a]):
            if len(match_loop)==0:
                assert idx==1 # default pair
                continue
            # matches_dict={}
            match_file=Path(os.path.join(self.opt.env.workspace_dir, f'match_tg_{idx}.h5'))
            match_file_writer = h5py.File(match_file, 'w')
            pairs_file=Path(os.path.join(self.opt.env.workspace_dir, f'pairs_tg_{idx}.txt'))
            feature_output_path=Path(self.opt.env.workspace_dir,f'keypoints_tg_{idx}.h5')
            
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
            kps_new,ka_data,feature_manager,track_labels,score_labels,root_labels,graph=refiner.refine_keypoints_and_get_track(output_path=feature_output_path,
                                             features_path=feature_path,
                                             image_dir=Path(example_path).parent,
                                             pairs_path=pairs_file,
                                             matches_path=match_file
                                             )
            for k in kps_new.keys():
                kps_new[k]/=scaling_kp
            self.track_graph.append(TrackGraph(graph,track_labels,score_labels,root_labels,self.image_name_to_id,kps_new,self.device,W,H,cfg={}))
            
            del track_labels,score_labels,root_labels,graph

        if self.opt.track.precompute_ka_for_track_loss:
            export_trackgraph(self.track_graph,precomputed_path)
            
            
            
            """
            Things we need to save: 
                after get keypoints, save new keypoints and TrackGraph object to fixed data path
                data/precomputed_tracks/datasets/scan_name, so we don't need pix-sfm and colmap
                
                self.opt.track.enable_ka_for_track_loss: not changed
                self.opt.track.enable_ka_for_corrs_loss: with postfix corrs ? 
                    Ideally it should be the same; but using different filename to avoid further change
                self.opt.track.precompute_ka_for_track_loss: get and save; 
                self.opt.track.precompute_ka_for_corrs_loss:
                self.opt.env.precomputed:
                We can add temporal break with both loss on now, just save the files on kw60746
                
            """
            
            # * do we need for loops here ??? early return !
            # for pair in match_loop: 
            #     # matches_dict[names_to_pair(image_names[pair[2]], image_names[pair[1]])] = matches
            #     id_in_flow_tensor, id_self, id_matching_view = pair[0], pair[1], pair[2]
            #     corres_map_self_to_other_ = self.corres_maps[id_in_flow_tensor].permute(1, 2, 0)[:, :, :2]  # (H, W, 2)
            #     conf_map_self_to_other_ = self.conf_maps[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
            #     variance_self_to_other_ = None
            #     mask_correct_corr = self.mask_valid_corr[id_in_flow_tensor].permute(1, 2, 0)  # (H, W, 1)
                
            #     # for correspondence 
            #     corres_map_self_to_other = corres_map_self_to_other_.detach() # [h,w,2], pixel to pixel matching
            #     conf_map_self_to_other = conf_map_self_to_other_.detach() # [h,w,1]
            #     mask_correct_corr = mask_correct_corr.detach().squeeze(-1)  # (H, W)
            #     corres_map_self_to_other_rounded = torch.round(corres_map_self_to_other).long()  # (H, W, 2)
            #     corres_map_self_to_other_rounded_flat = \
            #         corres_map_self_to_other_rounded[:, :, 1] * W + corres_map_self_to_other_rounded[:, :, 0] # corresponds to index in flattedned array (in H*W)
                
                
            #     pixels_in_self = self.grid[mask_correct_corr].cpu().numpy()  # [N_ray, 2], absolute pixel locations of correct
            #     pixels_in_other = corres_map_self_to_other[mask_correct_corr].cpu().numpy() # [N_ray, 2], absolute pixel locations, float
            #     conf_values = conf_map_self_to_other[mask_correct_corr]  # [N_ray, 1] 
            #     # grid: (h,w,2), element: (idx_w,idx_h)
            #     pixels_in_self_rounded=np.int32(np.round(pixels_in_self))
            #     pixels_in_other_rounded = np.int32(np.round(pixels_in_other))
            #     conf_values = conf_values.cpu().numpy()
                
            #     name_of_pair=names_to_pair(image_names[pair[1]], image_names[pair[2]])
            #     # XY matches 
            #     idx_A = (pixels_in_self_rounded[:, 1] * W + pixels_in_self_rounded[:, 0]).reshape(-1) # flatten img0 pixel index
            #     idx_B = (pixels_in_other_rounded[:, 1] * W + pixels_in_other_rounded[:, 0]).reshape(-1) # flatten img1 pixel index
                
            #     # * reload the optimized keypoints back to memory
            #     source_kps,target_kps=kps_new[image_names[pair[1]]],kps_new[image_names[pair[2]]]
            #     pixels_in_self_opt,pixels_in_other_opt=source_kps[idx_A],target_kps[idx_B]
            #     pixels_in_self_opt_,pixels_in_other_opt_=torch.from_numpy(pixels_in_self_opt).to(self.device),torch.from_numpy(pixels_in_other_opt).to(self.device)
                
            #     self.pixels_map[id_in_flow_tensor][0]=pixels_in_self_opt_.float()
            #     self.pixels_map[id_in_flow_tensor][1]=pixels_in_other_opt_.float()
            
            # conversion between kps_new and pixel coordinates
            #       Known pixel coordinate: feature_idx = y * W + x; kps_new[feature_idx]=(x_delta,y_delta)
                
        return
