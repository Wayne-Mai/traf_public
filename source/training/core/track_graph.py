import numpy as np
import torch
from easydict import EasyDict as edict
from typing import Any, Dict, Tuple
from omegaconf import OmegaConf
import random
from copy import deepcopy


def override_options(opt,opt_over,key_stack=[],safe_check=False):
    """Overrides edict with a new edict. """
    for key,value in opt_over.items():
        if isinstance(value,dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key,dict()),value,key_stack=key_stack+[key],safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y","n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new=="n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt


class Node:
    def __init__(self,idx,image_id,feature_idx,is_root,track_idx,score):
        self.node_idx=idx
        self.image_id=image_id
        # self.image_name=image_name
        self.feature_idx=feature_idx
        self.is_root=is_root
        self.track_idx=track_idx
        self.score=score
        # should we record score from match 1 to match 2 ?
        # the score represents the sum of total edges connected to this node, whether in or out
        # i think this is more reasonable
        # let's see what can be done within this node score

class Track:
    def __init__(self,track_label):
        self.len=0
        self.root=None
        self.nodes=[]
        self.label=track_label
    
    def add_node(self,node_py:Node):
        self.nodes.append(node_py)
        if node_py.is_root:
            self.root=self.len
        self.len+=1


class TrackGraph:
    default_conf = edict({
        'num_ray':2048,
        'sample_strategy':'long_track_first',
        'tolerance':10
    })
    
    
    def __init__(self,graph,track_labels,score_labels,root_labels,name_to_gid,kps_new,device,W,H,cfg={}):
        # name_to_gid: global image id match with data loader
        self.conf=override_options(self.default_conf, cfg)
        self.nodes=[None]*len(graph.nodes)
        self.num_tracks=np.max(track_labels)+1
        self.keypoints={name_to_gid[k]:v  for k,v in kps_new.items()}
        self.device=device
        # self.image_name_to_gid=deepcopy(name_to_gid)
        # labels start from 0
        self.tracks=[Track(idx) for idx in range(self.num_tracks)]
        
        for node in graph.nodes:
            image_name = graph.image_id_to_name[node.image_id]
            g_img_id=name_to_gid[image_name]
            
            node_py=Node(
                idx=node.node_idx,
                image_id=g_img_id,
                # image_name=image_name,
                feature_idx=node.feature_idx,
                is_root=root_labels[node.node_idx],
                track_idx=track_labels[node.node_idx],
                score=score_labels[node.node_idx]
            )
            self.nodes[node.node_idx]=node_py
            self.tracks[track_labels[node.node_idx]].add_node(node_py)
        # {1: 754, 2: 7061, 3: 1942}, why we have track len==1 ?
        self.track_lens=[x.len for x in self.tracks]
        
        filtered_tracks=[]
        for track in self.tracks:
            if track.len > 1:
                filtered_tracks.append(track)
        self.tracks=filtered_tracks
        self.track_lens=[x.len for x in self.tracks]
        self.num_tracks=len(self.tracks)
        numbers = np.array(self.track_lens).astype(float)
        total_sum = numbers.sum()
        probabilities = numbers / total_sum
        self.expected_len = (probabilities * numbers).sum()
        
        # init enough track variable
        self.enough_track=True
        self.W=W
        self.H=H
        self.sample_tracks()
    
    # note : can also be warped into pytorch data loader and sampler, 
    # using naive python to implement for convenience now
    # actually it's okay to slightly over 1024 since we are doing staged rendering with reference to pixels
    def sample_tracks(self):
        if self.conf.sample_strategy=='long_track_first':
            if self.default_conf.num_ray>len(self.nodes) and self.enough_track:
                print(f"Warning : no enough rays for {self.default_conf.num_ray},\
                      using {len(self.nodes)} rays instead")
                self.enough_track=False
            sampled_ray_num=min(self.default_conf.num_ray,len(self.nodes))
            sampled_track_num=int(sampled_ray_num/self.expected_len) 
            
            if sampled_track_num>self.num_tracks:
                if self.enough_track:
                    print(f"Warning : Can't sample {sampled_track_num} tracks from {self.num_tracks}, reducing num tracks ")
                    self.enough_track=False
                sampled_track_num=self.num_tracks
                
            # expected_len: 2.2; sampled_track_num: 912;
            selected_tracks=random.sample(self.tracks, sampled_track_num)
            return selected_tracks
                
        else:
            raise NotImplementedError()
        
    def tracks_to_image_pixels(self,selected_tracks):
        # should be called after construction of track graph and training, return dict:  {image_name: pixels}
        """
            but how to reserve the structure for loss calculation ? 
            we need a function: (track_idx): corresponding idx to find the pixel
            
            return: {image_id: pixels_in_2d} [N,2]
            record: track_idx -> node_idx, given node_idx, we need image_id and N_idx in todo pixels of this image
            
            300*400=12000
            for debug: node_idx: 2520; feature idx: 48194; keypoints[0][48194]: (194,120); 120*400+194=48194
        """
        todo_pixels_dict={}
        node_record_dict={}
        track_label_dict=[]
        for track in selected_tracks:
            assert track.label not in track_label_dict
            track_label_dict.append(track.label)
            for node in track.nodes:
                if node.image_id not in todo_pixels_dict:
                    todo_pixels_dict[node.image_id]=[self.keypoints[node.image_id][node.feature_idx]]
                else:
                    todo_pixels_dict[node.image_id].append(self.keypoints[node.image_id][node.feature_idx])
                assert  node.node_idx not in node_record_dict
                node_record_dict[node.node_idx]=(node.image_id,len(todo_pixels_dict[node.image_id])-1)
        
        todo_pixels_dict={k:torch.tensor(np.array(v),dtype=torch.float32).to(self.device) for k,v in todo_pixels_dict.items()}
        return todo_pixels_dict,node_record_dict
        # {num_img: [(x,y),...,]}; record_dict{num_img: (img_id,n)}
        
        
    def get_pixels_for_sampled_tracks(self,sampled_tracks,to_render_ray_int_dict,node_record_dict,ret_dict,images):
        """
        elements we need in the returned dict:
            ret_self.rgb.view(-1, 3)
            ret_self.rgb_fine
            ret_other.depth
            ret_self.depth_fine
            
            reminder: should every track contribute equally to the loss ?
                      how about contribution loss ? the confidence should it related to each pixel ?
                      what does pixel score come from? PixelSfM or MatchNet ? 
        
        """
        
        all_track_colors,all_track_fine_colors=[],[]
        all_gt_colors=[]
        all_root_idx=[]
        all_track_depths,all_track_fine_depths=[],[] 
        
        for track in sampled_tracks:
            track_colors,track_fine_colors=[],[]
            track_depths,track_fine_depths=[],[]
            gt_colors=[]
            for node in track.nodes:
                node_img_id,node_idx_in_todo=node_record_dict[node.node_idx]
                # node_ray_idx=to_render_ray_int_dict[node_img_id][node_idx_in_todo]
                node_color=ret_dict[node_img_id].rgb.view(-1, 3)[node_idx_in_todo]
                gt_color=images[node_img_id].view(-1, 3)[node_idx_in_todo]
                track_depth=ret_dict[node_img_id].depth[0,node_idx_in_todo,0]
                track_depths.append(track_depth)
                track_colors.append(node_color)
                gt_colors.append(gt_color)
                if 'rgb_fine' in ret_dict[node_img_id].keys():
                    node_fine_color=ret_dict[node_img_id].rgb_fine.view(-1, 3)[node_idx_in_todo]
                    track_fine_colors.append(node_fine_color)
                if 'depth_fine' in ret_dict[node_img_id].keys():
                    track_fine_depth=ret_dict[node_img_id].depth[0,node_idx_in_todo,0]
                    track_fine_depths.append(track_fine_depth)
            if len(track_fine_colors)>0:
                track_fine_colors=torch.stack(track_fine_colors)
                all_track_fine_colors.append(track_fine_colors)
            if len(track_fine_depths)>0:
                track_fine_depths=torch.stack(track_fine_depths)
                all_track_fine_depths.append(track_fine_depths)
            
            # todo: make it continguous
            track_colors=torch.stack(track_colors)
            track_depths=torch.stack(track_depths)
            gt_colors=torch.stack(gt_colors)
            all_track_colors.append(track_colors)
            all_gt_colors.append(gt_colors)
            all_root_idx.append(track.root)
            all_track_depths.append(track_depths)
            
        rendered_dict={
            'all_track_colors':all_track_colors,
            'all_gt_colors':all_gt_colors,
            'all_root_idx':all_root_idx,
            'all_track_depths':all_track_depths
        }
        if 'rgb_fine' in ret_dict[node_img_id].keys():
            rendered_dict['all_track_fine_colors']=all_track_fine_colors
        if 'depth_fine' in ret_dict[node_img_id].keys():
            rendered_dict['all_track_fine_depths']=all_track_fine_colors
        
        
        return rendered_dict
        # (num_sampled_track,len_track,pixel_rgb_3)
    
  
    
    def img_id_feature_idx_to_pixel_loc(self,img_id,feature_idx):
        # Q: the feature idx starts from all or just has match ?
        # A: now we use node_record_dict, discard this function now
        return self.keypoints[img_id][feature_idx]
    
import os,pickle
from pathlib import Path

def export_trackgraph(track_graph,precomputed_path):
    Path(os.path.dirname(precomputed_path)).mkdir(exist_ok=True,parents=True)
    with open(precomputed_path, 'wb') as file:
        pickle.dump(track_graph, file)
    print(f"Precomputed track graph from {precomputed_path} saved.")