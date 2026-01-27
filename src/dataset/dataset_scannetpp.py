


import random

import numpy as np
import PIL
import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset

from dataclasses import dataclass


from .scannetpp_utils.image import ImgNorm
from .scannetpp_utils import cropping as cropping
from .scannetpp_utils.geometry import depthmap_to_absolute_camera_coordinates

from .dataset import DatasetCfgCommon
from pathlib import Path
from typing import Literal, List
from .types import Stage
from .view_sampler import ViewSampler

import os
import re
import json

from torch import Tensor
from jaxtyping import Float
from einops import repeat
from PIL import Image
import re
from torchvision.transforms import PILToTensor

@dataclass
class DatasetScanNetPPCfg(DatasetCfgCommon):
    name: Literal["scannetpp"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    near: float = 0.5
    far: float = 15.0

class DatasetScanNetPP(Dataset):

    def __init__(
        self,
        cfg: DatasetScanNetPPCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ):

        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.pil_to_tensor = PILToTensor()


        self.root = cfg.roots[0]
        self.sequences = []
        sequences = os.listdir(os.path.join(self.root, 'scenes'))   

        ############################################
        sequences = [seq for seq in sequences if os.path.isdir(os.path.join(self.root, 'scenes', seq, 'instance_segmentation'))]            
        P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
            ).astype(np.float32)
        for sequence in sequences:            
            input_processed_folder = os.path.join(self.root, 'scenes',  sequence)
            metadata_path = os.path.join(input_processed_folder, "transforms_train.json")            

            with open(metadata_path, "r") as f:
                data = json.load(f)

                # Get the intrinsics data for the frame
                K = np.eye(4, dtype=np.float32)
                K[0, 0] = data.get("fl_x")
                K[1, 1] = data.get("fl_y")
                K[0, 2] = data.get("cx")
                K[1, 2] = data.get("cy")
                frames = []
                for frame in data.get("frames", []):
                    c2w = np.array(frame.get("transform_matrix"), dtype=np.float32)
                    c2w = P @ c2w @ P.T

                    match = re.search(r'\d+', frame.get("file_path"))
                    index = int(match.group()) if match else -1  # Use -1 for missing index

                    if os.path.exists(os.path.join(input_processed_folder, 'images', frame.get("file_path"))):
                        frames.append({
                            "file_path": frame.get("file_path"),
                            "extrinsics": c2w,
                            "intrinsics": K.copy(),
                            "is_bad": frame.get("is_bad"),
                            "path_index": index,
                        })
                    else:
                        continue

                # Sort frames by path_index
                frames = sorted(frames, key=lambda x: x["path_index"])

                # Now build your lists in sorted order
                paths = [f["file_path"] for f in frames]
                extrinsics = [f["extrinsics"] for f in frames]
                intrinsics = [f["intrinsics"] for f in frames]
                is_bad = [f["is_bad"] for f in frames]
                path_index = [f["path_index"] for f in frames]

                self.sequences.append(
                    {
                        'sequence': sequence,
                        'paths': paths,
                        'extrinsics': np.stack(extrinsics),
                        'intrinsics': np.stack(intrinsics)[:, :3, :3],  
                        'is_bad': np.array(is_bad),
                        'path_index': path_index,
                    }
                )                                
        

    def lerp(self, initial, final):
        fraction = self.view_sampler.global_step / self.view_sampler.cfg.warm_up_steps
        return max(initial + (final - initial) * fraction, final)




    def __getitem__(self, index):

        while True:            
            sequence_dict = self.sequences[index]
            scene = sequence_dict['sequence']
            seg_dir = os.path.join(self.root, 'scenes', scene, 'instance_segmentation')            
            sequence_dir = os.path.join(self.cfg.roots[0], 'scenes', scene, 'images')
            paths = sequence_dict['paths']               
            if len(paths) < 2:
                raise ValueError(f"Not enough frames in sequence {sequence_dir} for training.")
            imshape = self.to_tensor(Image.open(os.path.join(sequence_dir, paths[0]))).shape
            extrinsics = torch.from_numpy(sequence_dict['extrinsics'].copy())
            intrinsics = torch.from_numpy(sequence_dict['intrinsics'].copy())
            is_bad = sequence_dict['is_bad']
            path_index = sequence_dict['path_index']
            
            context_index, target_indices = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
            if context_index[1]-context_index[0] != path_index[context_index[1]]-path_index[context_index[0]]:            
                continue
            

            intrinsics[:, :1] /= imshape[2]
            intrinsics[:, 1:2] /= imshape[1]

            example = {'scene': scene}

            context_images = []    
            context_segmentation = []        
            for idx in context_index:                
                img = Image.open(os.path.join(sequence_dir, paths[idx]))
                img = self.to_tensor(img.resize(self.cfg.image_shape[::-1]))
                context_images.append(img[None], )
                
                seg = Image.open(os.path.join(seg_dir, f"{paths[idx]}.png"))                                     
                seg = self.pil_to_tensor(seg.resize(self.cfg.image_shape[::-1], resample=Image.NEAREST))        
                context_segmentation.append(seg)
            context_images = torch.cat(context_images)    
            context_segmentation = torch.cat(context_segmentation, dim=0)        
            content = {"extrinsics": extrinsics[context_index],
                        "intrinsics": intrinsics[context_index],
                        "image": context_images,
                        "segmentation": context_segmentation,
                        "near": self.get_bound("near", len(context_index)),
                        "far": self.get_bound("far", len(context_index)),
                        "index": context_index,
                        }
            example['context'] = content
            target_images = []
            target_segmentation = []
            for idx in target_indices:
                img = Image.open(os.path.join(sequence_dir, paths[idx]))
                img = self.to_tensor(img.resize(self.cfg.image_shape[::-1]))
                target_images.append(img[None])                       
                seg = Image.open(os.path.join(seg_dir, f"{paths[idx]}.png"))
                seg = self.pil_to_tensor(seg.resize(self.cfg.image_shape[::-1], resample=Image.NEAREST))        
                target_segmentation.append(seg)
            target_images = torch.cat(target_images)
            target_segmentation = torch.cat(target_segmentation, dim=0)        
            example["target"] = {
                "extrinsics": extrinsics[target_indices],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "segmentation": target_segmentation,
                "near": self.get_bound("near", len(target_indices)),
                "far": self.get_bound("far", len(target_indices)),
                "index": target_indices,                    
            }

            return example
            
    
    def __iter__(self):
        """
        Returns an iterator that yields elements from the dataset.
        """
        for i in range(len(self)):
            yield self[i]

    def convert_poses(
            self,
            extrinsics,
            intrinsics,
    ):
        # If intrinsics is a numpy array, convert it to a torch tensor
        if isinstance(intrinsics, np.ndarray):
            intrinsics = torch.from_numpy(intrinsics)

        # If extrinsics is a list of numpy arrays, convert each element to a torch tensor
        if isinstance(extrinsics, list):
            extrinsics = [torch.from_numpy(e) if isinstance(e, np.ndarray) else e for e in extrinsics]

        new_intrinsics = intrinsics.clone()

        # Normalize the intrinsics
        IMAGE_SHAPE = (int(1168/4), int(1752/4))
        new_intrinsics[0] /= IMAGE_SHAPE[1]
        new_intrinsics[1] /= IMAGE_SHAPE[0]
        new_intrinsics = new_intrinsics[:3, :3]

        new_extrinsics = torch.stack(extrinsics, dim=0)
        new_intrinsics = repeat(new_intrinsics, "h w -> b h w", b=new_extrinsics.shape[0]).clone()

        return new_extrinsics, new_intrinsics



    def convert_images(
        self,
        images,
    ):
        torch_images = []
        for image_path in images:
            image = Image.open(image_path)
            image = self.to_tensor(image)
            torch_images.append(image)
            assert image.shape == (3, int(1168/4), int(1752/4))
        return torch.stack(torch_images)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)
    
    def __len__(self):        
        return len(self.sequences) 
        
