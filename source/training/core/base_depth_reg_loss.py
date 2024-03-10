
import torch
from easydict import EasyDict as edict
import lpips
import torch.nn as nn
import numpy as np

from typing import Callable, Sequence, List, Mapping, MutableMapping, Tuple, Union, Dict
from typing import Any, Optional
from source.training.core.regularization_losses import depth_patch_loss, lossfun_distortion
from source.training.core.base_losses import BaseLoss

def get_smooth_loss(disp, img):
    """
    Code for GeoNet: Unsupervised Learning of Dense Depth, 
    Optical Flow and Camera Pose (CVPR 2018)
    
    
   
        
    """
    def gradient_x(img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy
    
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)
    
    
    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 3, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 3, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y

    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def sample_rays_for_patch(H: int, W: int, patch_size: int, precrop_frac: float=0.5, 
                          fraction_in_center: float=0., nbr: int=None
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Samples pixels/rays with patch formatting, ie the output shape 
    is (N, patch_size**2, 2)/(N, patch_size**2)"""

    # patch
    y_range = torch.arange(patch_size, dtype=torch.long)
    x_range = torch.arange(patch_size, dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [patch_size,patch_size]
    dxdy = torch.stack([X,Y],dim=-1).view(-1,2) # [patch_size**2,2]

    # all pixels
    y_range = torch.arange(H-(patch_size - 1),dtype=torch.long)
    x_range = torch.arange(W-(patch_size - 1),dtype=torch.long)
    Y,X = torch.meshgrid(y_range,x_range) # [H,W]
    xy_grid = torch.stack([X,Y],dim=-1).view(-1,2).long() # [HW,2]
    n = xy_grid.shape[0]
    x_ind = xy_grid[..., 0]
    y_ind = xy_grid[..., 1]

    if fraction_in_center > 0.:
        dH = int(H//2 * precrop_frac)
        dW = int(W//2 * precrop_frac)
        Y, X = torch.meshgrid(
                torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
            )
        pixels_in_center = torch.stack([X, Y], -1).view(-1, 2)  # [N, 2]
        if nbr is not None:
            nbr_center = int(nbr*fraction_in_center)
            nbr_all = nbr - nbr_center
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr_all]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]

            idx = torch.randperm(len(pixels_in_center), device=x_ind.device)[:nbr_center]
            pixels_in_center = pixels_in_center[idx] # [N, 2]
            x_ind = torch.cat((x_ind, pixels_in_center[..., 0]))
            y_ind = torch.cat((y_ind, pixels_in_center[..., 1]))  # 
    else:
        if nbr is not None:
            # select a subset of those
            idx = torch.randperm(len(x_ind), device=x_ind.device)[:nbr]
            x_ind = x_ind[idx]
            y_ind = y_ind[idx]
            
    n = len(x_ind)

    x_ind = (x_ind[:, None].repeat(1, patch_size**2) + dxdy[:, 0]).reshape(-1)
    y_ind = (y_ind[:, None].repeat(1, patch_size**2) + dxdy[:, 1]).reshape(-1)

    pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(n, patch_size**2, -1)

    rays = pixel_coords[..., 1] * W + pixel_coords[..., 0]

    return pixel_coords, rays  # (N, patch_size**2, 2), (N, patch_size**2)



# seems like this works with patch .....
class BaseDepthReguLoss(BaseLoss):
    """Class responsable for computing the photometric loss (Huber or MSE), the mask loss 
    along with typical regularization losses (depth patch, distortion..). """
    def __init__(self, opt: Dict[str, Any], nerf_net: torch.nn.Module, 
                 train_data: Dict[str, Any], device: torch.device):
        super().__init__(device)
        self.opt = opt
        self.device = device
        self.net = nerf_net
        self.train_data = train_data
        
        
    def compute_pixel_coords_for_patch(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """Compute pixel coords for a patch."""
        patch_size = self.opt.depth_reg.depth_regu_patch_size
        shape_ = pixel_coords.shape[:-1]
        x_ind = pixel_coords.view(-1, 2)[..., 0]
        y_ind = pixel_coords.view(-1, 2)[..., 1]


        x_ind = (x_ind[:, None].repeat(1, patch_size**2) + self.dxdy[:, 0]).reshape(-1)
        y_ind = (y_ind[:, None].repeat(1, patch_size**2) + self.dxdy[:, 1]).reshape(-1)

        pixel_coords = torch.stack([x_ind, y_ind], dim=-1).reshape(shape_ + (patch_size**2, -1))
        return pixel_coords
    
    def sample_path(self):
        
        
        pass

    def compute_loss(self, opt: Dict[str, Any], data_dict: Dict[str, Any], output_dict: Dict[str, Any], 
                     iteration: int, mode: str=None, plot: bool=False, **kwargs
                     ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Args:
            opt (edict): settings
            data_dict (edict): Input data dict. Contains important fields:
                            - Image: GT images, (B, 3, H, W)
                            - intr: intrinsics (B, 3, 3)
                            - idx: idx of the images (B)
                            - depth_gt (optional): gt depth, (B, 1, H, W)
                            - valid_depth_gt (optional): (B, 1, H, W)
            output_dict (edict): Output dict from the renderer. Contains important fields
                                - idx_img_rendered: idx of the images rendered (B), useful 
                                in case you only did rendering of a subset
                                - ray_idx: idx of the rays rendered, either (B, N) or (N)
                                - rgb: rendered rgb at rays, shape (B, N, 3)
                                - depth: rendered depth at rays, shape (B, N, 1)
                                - rgb_fine: rendered rgb at rays from fine MLP, if applicable, shape (B, N, 3)
                                - depth_fine: rendered depth at rays from fine MLP, if applicable, shape (B, N, 1)
            iteration (int)
            mode (str, optional): Defaults to None.
            plot (bool, optional): Defaults to False.
        """
        
        stats_dict, plotting_dict, loss_dict = {}, {}, {'depth_reg': torch.tensor(0., requires_grad=True).to(self.device)}
        
        if opt.model=='nerf_gt_poses':
            pass
        else:
            start_iter_depth_cons = opt.depth_reg.start_iter_ratio * opt.max_iter 
            if iteration < start_iter_depth_cons:
                return loss_dict, stats_dict, plotting_dict
        
        patch_size=self.opt.depth_reg.depth_regu_patch_size
        patch_num=self.opt.depth_reg.patch_num
        
        
        B, _, H, W = data_dict.image.shape
        images = data_dict.image # (B, 3, H, W) 
        
        sampled_img_idx=np.random.randint(B)
        sampled_img_idx=[sampled_img_idx]*patch_num
        sampled_img=images[sampled_img_idx]
        
        # (N_patches, patch_size**2, 2), (N_patches, patch_size**2)
        pixel_coords,rays_idx=sample_rays_for_patch(H=H, W=W, patch_size=patch_size, precrop_frac=0.5, 
                          fraction_in_center=0., nbr=patch_num)
        
        # since we don't give specific img idx, will render this sampled patches for all sparse view image
        # but it's better to do it for one image, since we have up to 9 sparse views, gpu may oom
        
        # rays_idx=rays_idx.reshape(-1,1)
        # (N, patch_size**2) -> ray_idx: (2048/num_images,1); 
        ret_dict = self.net.render_image_at_specific_rays(self.opt, data_dict, img_idx=sampled_img_idx,
                                                             ray_idx=rays_idx, iter=iteration, mode="train")
        # ret_dict shape: usually (batch,N,3) accepted ray_idx: (b,n) or n
        # get_smooth_loss(disp, img), b,h,w,3 and b,h,w,1
        
        depth=ret_dict.depth.reshape(patch_num,patch_size,patch_size,-1)
        if self.opt.depth_reg.color=='gt':
            patches_index=pixel_coords.reshape(patch_num,patch_size,patch_size,2)
            pixel_patch_values = sampled_img[0, :, patches_index[:, :, :, 1], patches_index[:, :, :, 0]]
            # 3,n,size,size -> n,size,size,3
            color = pixel_patch_values.permute(1,2,3,0)
            
        elif self.opt.depth_reg.color=='render':
            color=ret_dict.rgb.reshape(patch_num,patch_size,patch_size,-1)
        else:
            raise NotImplementedError("Invalid color supervision")
        
        loss_dict['depth_reg']+=get_smooth_loss(1./depth,color)
        
        # * consider about weight,  we don't use huber loss because we want to penalize outlier more
        if 'depth_fine' in ret_dict.keys():
            depth_fine=ret_dict.depth_fine.reshape(patch_num,patch_size,patch_size,-1)
            if self.opt.depth_reg.color=='gt':
                color_fine=color
            else:
                color_fine=ret_dict.rgb_fine.reshape(patch_num,patch_size,patch_size,-1)
            loss_dict['depth_reg']+=get_smooth_loss(1./depth_fine,color_fine)
            loss_dict['depth_reg']/=2.
        
        
        
        return loss_dict, stats_dict, plotting_dict
    
    
