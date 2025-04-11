import random
import numpy as np
import torch
from typing import List

from utils3D.lossandmetrics import bbox_iov


def tensor_cutout(im: torch.Tensor, labels, cutout_params: List[List[float]], p=0.5):
    """Applies image cutout augmentation https://arxiv.org/abs/1708.04552

    Args:
        im (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): Med YOLO labels corresponding to im. class z1 x1 y1 z2 x2 y2
        cutout_params (List[List[float]]): a list of ordered pairs that set sizes and numbers of cutout boxes.
                                           the maximum extent of the boxes is set by the first element of the ordered pair
                                           the number of boxes with that maximum extent is set by the second element of the ordered pair
        p (float, optional): probability of performing cutout augmentation. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    if random.random() < p:
        d, h, w = im.shape[1:]
        scales = []
        for param_pair in cutout_params:
            scales = scales + [param_pair[0]]*param_pair[1]
       
        for s in scales:
            mask_d = random.randint(1, int(d * s))  # create random masks
            mask_h = random.randint(1, int(h * s))
            mask_w = random.randint(1, int(w * s))
            
            # box
            zmin = max(0, random.randint(0, d) - mask_d // 2)
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            zmax = min(d, zmin + mask_d)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)
            
            # apply random greyscale mask
            # images scaled between 0 and 1 after being returned by the dataset
            im[:,zmin:zmax, ymin:ymax, xmin:xmax] = random.uniform(torch.min(im), torch.max(im))
            
            # remove obscured labels
            if len(labels) and s > 0.03:
                box = np.array([zmin, xmin, ymin, zmax, xmax, ymax], dtype=np.float32)
                iov = bbox_iov(box, labels[:, 1:7])  # intersection over volume
                labels = labels[iov < 0.60]  # remove >60% obscured labels
                
    return im, labels


def random_zoom(image: torch.Tensor, labels, max_zoom=1.5, min_zoom=0.7, p=0.5):
    """Randomly zooms in or out of a random part of the input image.

    Args:
        image (torch.Tensor): 3-D tensor to be augmented.
        labels (List[float]): MedYOLO labels corresponding to im. class z1 x1 y1 z2 x2 y2
        max_zoom (float, optional): maximum edge length multiplier. Defaults to 1.5.
        min_zoom (_type_, optional): minimum edge length multiplier. Defaults to 0.7.
        p (float, optional): probability of zooming the input. Defaults to 0.5.

    Returns:
        im: Augmented tensor.
        y: Adjusted labels.
    """
    
    augmented_labels = labels.clone() if isinstance(labels, torch.Tensor) else np.copy(labels)
    
    if random.random() < p:
        # retrieve original image shape (this is resized to imgsz_z x imgsz_y x imgsz_x by this point in the dataloader)
        channels, d, h, w = image.shape
        
        zoom_factor = random.uniform(min_zoom, max_zoom)

        # add batch dimension for functional interpolate
        image = torch.unsqueeze(image, 0)
        image = torch.nn.functional.interpolate(image, scale_factor=zoom_factor, mode='trilinear', align_corners=False)
        # remove batch dimension for compatibility with later code
        image = torch.squeeze(image, 0)

        # retrieve new image shape
        new_d, new_h, new_w = image.shape[1:]
        
        # shrink/expand labels
        augmented_labels[:, 1:7] = augmented_labels[:, 1:7]*zoom_factor
        
        # crop/pad the zoomed image back to the input size and position it randomly relative to the new tensor
        if zoom_factor >= 1.:
            # new side lengths longer than original side lengths
            # crop to original im.shape (center needs at least original_length/2 distance to each edge to preserve original image)
            zoom_center_d = random.randint(d//2, new_d-d//2)
            zoom_center_h = random.randint(h//2, new_h-h//2)
            zoom_center_w = random.randint(w//2, new_w-w//2)
            
            zmin = zoom_center_d - d//2
            xmin = zoom_center_w - w//2
            ymin = zoom_center_h - h//2
            zmax = zmin + d
            xmax = xmin + w
            ymax = ymin + h
            
            image = image[:,zmin:zmax, ymin:ymax, xmin:xmax]

            # move labels to correspond to new center of zoom
            augmented_labels[:, 1] = augmented_labels[:, 1] - zmin
            augmented_labels[:, 4] = augmented_labels[:, 4] - zmin
            augmented_labels[:, 2] = augmented_labels[:, 2] - xmin
            augmented_labels[:, 5] = augmented_labels[:, 5] - xmin
            augmented_labels[:, 3] = augmented_labels[:, 3] - ymin
            augmented_labels[:, 6] = augmented_labels[:, 6] - ymin
            
            # crop labels beyond bounds of new image
            if isinstance(augmented_labels, torch.Tensor):  # faster individually
                augmented_labels[:, 1].clamp_(0, d)  # z1
                augmented_labels[:, 2].clamp_(0, w)  # x1
                augmented_labels[:, 3].clamp_(0, h)  # y1
                augmented_labels[:, 4].clamp_(0, d)  # z2
                augmented_labels[:, 5].clamp_(0, w)  # x2
                augmented_labels[:, 6].clamp_(0, h)  # y2
            else:  # np.array (faster grouped)
                augmented_labels[:, [1, 4]] = augmented_labels[:, [1, 4]].clip(0, d)  # z1, z2
                augmented_labels[:, [2, 5]] = augmented_labels[:, [2, 5]].clip(0, w)  # x1, x2
                augmented_labels[:, [3, 6]] = augmented_labels[:, [3, 6]].clip(0, h)  # y1, y2
             
        else:
            # new side lengths shorter than original side lengths
            # pad to original image shape
            zoom_center_d = random.randint(new_d//2 + 1, d-new_d//2 - 1)
            zoom_center_h = random.randint(new_h//2 + 1, h-new_h//2 - 1)
            zoom_center_w = random.randint(new_w//2 + 1, w-new_w//2 - 1)
                
            zmin = zoom_center_d - new_d//2
            xmin = zoom_center_w - new_w//2
            ymin = zoom_center_h - new_h//2
            
            zmax = zmin + new_d
            xmax = xmin + new_w
            ymax = ymin + new_h
            
            # create a new tensor 
            new_image = torch.rand(channels, d, h, w)*(torch.max(image) - torch.min(image)) + torch.min(image)
            new_image[:,zmin:zmax, ymin:ymax, xmin:xmax] = image
            image = new_image
            del new_image
            
            # move labels to correspond to new center of zoom
            augmented_labels[:, 1] = augmented_labels[:, 1] + zmin
            augmented_labels[:, 4] = augmented_labels[:, 4] + zmin
            augmented_labels[:, 2] = augmented_labels[:, 2] + xmin
            augmented_labels[:, 5] = augmented_labels[:, 5] + xmin
            augmented_labels[:, 3] = augmented_labels[:, 3] + ymin
            augmented_labels[:, 6] = augmented_labels[:, 6] + ymin

    return image, augmented_labels
