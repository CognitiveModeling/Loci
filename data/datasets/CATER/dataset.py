from torch.utils import data
from typing import Tuple, Union, List
import numpy as np
import json
import math
import cv2
import h5py
import os
import pickle

__author__ = "Manuel Traub"

class RamImage():
    def __init__(self, path):
        
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()

        self.img_raw = np.frombuffer(img_str, np.uint8)

    def to_numpy(self):
        return cv2.imdecode(self.img_raw, cv2.IMREAD_COLOR) 

class CaterSample(data.Dataset):
    def __init__(self, root_path: str, data_path: str, size: Tuple[int, int]):

        data_path = os.path.join(root_path, data_path, "train", f'{size[0]}x{size[1]}')

        frames = []
        self.size = size

        for file in os.listdir(data_path):
            if file.startswith("frame") and (file.endswith(".jpg") or file.endswith(".png")):
                frames.append(os.path.join(data_path, file))

        frames.sort()
        self.imgs = []
        for path in frames:
            self.imgs.append(RamImage(path))

    def get_data(self):

        frames = np.zeros((301,3,self.size[1], self.size[0]),dtype=np.float32)
        for i in range(len(self.imgs)):
            img = self.imgs[i].to_numpy()
            frames[i] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return frames


class CaterDataset(data.Dataset):

    def save(self):
        state = { 'samples': self.samples, 'labels': self.labels }
        with open(self.file, "wb") as outfile:
    	    pickle.dump(state, outfile)

    def load(self):
        with open(self.file, "rb") as infile:
            state = pickle.load(infile)
            self.samples = state['samples']
            self.labels  = state['labels']

    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int]):

        data_path  = f'data/data/video/{dataset_name}'
        data_path  = os.path.join(root_path, data_path)
        self.file  = os.path.join(data_path, f'dataset-{size[0]}x{size[1]}-{type}.pickle')
        self.train = (type == "train")
        self.val   = (type == "val")
        self.test  = (type == "test")

        self.samples    = []
        self.labels     = []

        if os.path.exists(self.file):
            self.load()
        else:

            samples         = list(filter(lambda x: x.startswith("0"), next(os.walk(data_path))[1]))
            num_all_samples = len(samples)
            num_samples     = 0
            sample_start    = 0

            if type == "train":
                num_samples = int(num_all_samples * 0.7 * 0.8)
            if type == "val":
                num_samples = int(num_all_samples * 0.7 * 0.2)
            if type == "test":
                num_samples = int(num_all_samples * 0.3)

            if type == "val":
                sample_start = int(num_all_samples * 0.7 * 0.8)
            if type == "test":
                sample_start = int(num_all_samples * 0.7)

            for i, dir in enumerate(samples[sample_start:sample_start+num_samples]):
                self.samples.append(CaterSample(data_path, dir, size))
                self.labels.append(json.load(open(os.path.join(data_path, "labels", f"{dir}.json"))))

                print(f"Loading CATER {type} [{i * 100 / num_samples:.2f}]", flush=True)

            self.save()
        
        self.length     = len(self.samples)
        self.background = None
        if "background.jpg" in os.listdir(data_path):
            self.background = cv2.imread(os.path.join(data_path, "background.jpg"))
            self.background = cv2.resize(self.background, dsize=size, interpolation=cv2.INTER_CUBIC)
            self.background = self.background.transpose(2, 0, 1).astype(np.float32) / 255.0
            self.background = self.background.reshape(1, self.background.shape[0], self.background.shape[1], self.background.shape[2])

        print(f"CaterDataset[{type}]: {self.length}")

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')


        self.cam = np.array([
            (1.4503, 1.6376,  0.0000, -0.0251),
            (-1.0346, 0.9163,  2.5685,  0.0095),
            (-0.6606, 0.5850, -0.4748, 10.5666),
            (-0.6592, 0.5839, -0.4738, 10.7452)
        ])

        self.z = 0.3421497941017151

        self.object_actions = {
            'sphere_slide':        0,
            'sphere_pick_place':   1,
            'spl_slide':           2,
            'spl_pick_place':      3,
            'spl_rotate':          4,
            'cylinder_slide':      5,
            'cylinder_pick_place': 6,
            'cylinder_rotate':     7,
            'cube_slide':          8,
            'cube_pick_place':     9,
            'cube_rotate':        10,
            'cone_slide':         11,
            'cone_pick_place':    12,
            'cone_contain':       13,
            'sphere_no_op':       14,
            'spl_no_op':          14,
            'cylinder_no_op':     14,
            'cube_no_op':         14,
            'cone_no_op':         14,
        }

        self.object_materials = {
            'sphere_rubber':   0,
            'sphere_metal':    1,
            'cylinder_rubber': 2,
            'cylinder_metal':  3,
            'cube_rubber':     4,
            'cube_metal':      5,
            'cone_rubber':     6,
            'cone_metal':      7,
        }

        self.object_sizes = {
            'sphere_small':    0,
            'sphere_medium':   1,
            'sphere_large':    2,
            'cylinder_small':  3,
            'cylinder_medium': 4,
            'cylinder_large':  5,
            'cube_small':      6,
            'cube_medium':     7,
            'cube_large':      8,
            'cone_small':      9,
            'cone_medium':    10,
            'cone_large':     11,
        }

        self.object_colors = {
            'sphere_red':       0,
            'sphere_purple':    1,
            'sphere_yellow':    2,
            'sphere_brown':     3,
            'sphere_gray':      4,
            'sphere_blue':      5,
            'sphere_cyan':      6,
            'sphere_green':     7,
            'cylinder_red':     8,
            'cylinder_purple':  9,
            'cylinder_yellow': 10,
            'cylinder_brown':  11,
            'cylinder_gray':   12,
            'cylinder_blue':   13,
            'cylinder_cyan':   14,
            'cylinder_green':  15,
            'cube_red':        16,
            'cube_purple':     17,
            'cube_yellow':     18,
            'cube_brown':      19,
            'cube_gray':       20,
            'cube_blue':       21,
            'cube_cyan':       22,
            'cube_green':      23,
            'cone_red':        24,
            'cone_purple':     25,
            'cone_yellow':     26,
            'cone_brown':      27,
            'cone_gray':       28,
            'cone_blue':       29,
            'cone_cyan':       30,
            'cone_green':      31,
        }

    def project_3d_point(self, pts):
        """
        Args:     pts: Nx3 matrix, with the 3D coordinates of the points to convert
        Returns:  Nx2 matrix, with the coordinates of the point in 2D
        """
        p = np.matmul(
            self.cam,
            np.hstack((pts, np.ones((pts.shape[0], 1)))).transpose()).transpose()
        # The predictions are -1 to 1, Negating the 2nd to put low Y axis on top
        p[:, 0] /= p[:, -1]
        p[:, 1] /= -p[:, -1]
        return np.concatenate((p[:,1:2],p[:,0:1]), axis=1)

    def snitch_position(self, metadata):

        objects = metadata['objects']
        object = [el for el in objects if el['shape'] == 'spl'][0]
        pts = np.zeros((len(object['locations']), 3))
        for i in range(len(object['locations'])):
            pts[i] = object['locations'][str(i)] 

        return pts #self.project_3d_point(pts)

    def localize_label(self, metadata, num_rows=3, num_cols=3):

        objects = metadata['objects']
        object = [el for el in objects if el['shape'] == 'spl'][0]
        pos = object['locations'][str(len(object['locations']) - 1)]
        if num_rows != 3 or num_cols != 3:
            # In this case, need to scale the pos values to scale to the new num_rows etc
            pos[0] *= num_cols * 1.0 / 3
            pos[1] *= num_rows * 1.0 / 3
        # Without math.floor it would screw up on negative axis
        x, y = (int(math.floor(pos[0])) + num_cols,
                int(math.floor(pos[1])) + num_rows)
        cls_id = y * (2 * num_cols) + x

        return cls_id
        #return np.eye(num_rows * num_cols * 4)[cls_id]


    def visibility_mask(self, metadata):
        
        movements  = metadata['movements']
        objects    = metadata['objects']
        visible = {el['instance']: np.ones((301)) for el in objects}

        for name, motions in movements.items():
            if name.startswith('Cone_'):
                start  = -1
                end    = 301
                object = ""
                for motion in motions:
                    if motion[0] == '_contain':
                        start  = motion[3]
                        object = motion[1]
                    if motion[0] == '_pick_place' and start > 0:
                        end = motion[2]
                        visible[object][start:end] = 0
                        start = -1
                        end   = 301

                if start > 0:
                    visible[object][start:end] = 0

        return visible

    def actions_over_time(self, metadata):
        
        movements = metadata['movements']
        objects   = metadata['objects']
        to_type = {el['instance']: el['shape'] for el in objects}

        actions_visible = np.zeros((301, 15))
        actions_hidden  = np.zeros((301, 15))

        visible = self.visibility_mask(metadata)

        for name, motions in movements.items():
            for motion in motions:
                action_index = self.object_actions[to_type[name] + motion[0]]
                actions_visible[motion[2]:motion[3],action_index] += visible[name][motion[2]:motion[3]]
                actions_hidden[motion[2]:motion[3],action_index]  += 1 - visible[name][motion[2]:motion[3]]
        
        # remove no op 
        return actions_visible[:,:-1], actions_hidden[:,:-1]

    def snitch_contained(self, metadata):
        visible = self.visibility_mask(metadata)
        return 1 - visible['Spl_0']

    def materials_over_time(self, metadata):
        
        movements = metadata['movements']
        objects   = metadata['objects']
        objects   = {el['instance']: el['shape'] + "_" + el['material'] for el in objects}

        materials_visible = np.zeros((301, 8))
        materials_hidden  = np.zeros((301, 8))

        visible = self.visibility_mask(metadata)
        
        for instance, class_name in objects.items():
            if instance != 'Spl_0': 
                index = self.object_materials[class_name]
                materials_visible[:,index] += visible[instance]
                materials_hidden[:,index]  += 1 - visible[instance]
        
        return materials_visible, materials_hidden

    def sizes_over_time(self, metadata):
        
        movements = metadata['movements']
        objects   = metadata['objects']
        objects   = {el['instance']: el['shape'] + "_" + el['size'] for el in objects}

        size_visible = np.zeros((301, 12))
        size_hidden  = np.zeros((301, 12))

        visible = self.visibility_mask(metadata)
        
        for instance, class_name in objects.items():
            if instance != 'Spl_0': 
                index = self.object_sizes[class_name]
                size_visible[:,index] += visible[instance]
                size_hidden[:,index]  += 1 - visible[instance]
        
        return size_visible, size_hidden

    def colors_over_time(self, metadata):
        
        movements = metadata['movements']
        objects   = metadata['objects']
        objects   = {el['instance']: el['shape'] + "_" + el['color'] for el in objects}

        color_visible = np.zeros((301, 32))
        color_hidden  = np.zeros((301, 32))

        visible = self.visibility_mask(metadata)
        
        for instance, class_name in objects.items():
            if instance != 'Spl_0': 
                index = self.object_colors[class_name]
                color_visible[:,index] += visible[instance]
                color_hidden[:,index]  += 1 - visible[instance]
        
        return color_visible, color_hidden

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        label = self.labels[index]

        snitch_positions  = self.snitch_position(label)
        snitch_label      = self.localize_label(label)
        snitch_contained  = self.snitch_contained(label)

        actions_visible,   actions_hidden   = self.actions_over_time(label)
        materials_visible, materials_hidden = self.materials_over_time(label)
        sizes_visible,     sizes_hidden     = self.sizes_over_time(label)
        colors_visible,    colors_hidden    = self.colors_over_time(label)
        
        if self.background is not None:
            return (
                self.samples[index].get_data(),
                self.background,
                snitch_positions, 
                snitch_label,
                snitch_contained, 
                actions_visible, 
                actions_hidden,
                materials_visible,
                materials_hidden,
                sizes_visible,
                sizes_hidden,
                colors_visible,
                colors_hidden
            )

        return (
            self.samples[index].get_data(),
            self.background,
            snitch_positions, 
            snitch_label,
            snitch_contained, 
            actions_visible, 
            actions_hidden,
            materials_visible,
            materials_hidden,
            sizes_visible,
            sizes_hidden,
            colors_visible,
            colors_hidden
        )


class CaterLatentDataset(data.Dataset):
    def __init__(self, root_path: str, filename: str, type: str):
        self.type = type
        self.data_path = os.path.join(root_path, filename)

        self.dataset = h5py.File(self.data_path, 'r')

        self.length = len(self.dataset['train']["snitch_positions"]) + len(self.dataset['test']["snitch_positions"])

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')

        self.dataset.close()
        self.dataset = None

        self.cam = np.array([
            (1.4503, 1.6376,  0.0000, -0.0251),
            (-1.0346, 0.9163,  2.5685,  0.0095),
            (-0.6606, 0.5850, -0.4748, 10.5666),
            (-0.6592, 0.5839, -0.4738, 10.7452)
        ])

        self.z = 0.3421497941017151

    def __len__(self):
        if self.type == "train":
            return int(self.length * 0.7 * 0.8)
        
        if self.type == "val":
            return int(self.length * 0.7 * 0.2)

        return int(self.length * 0.3)

    def __getitem__(self, index: int):
        if self.dataset is None:
            self.dataset = h5py.File(self.data_path, 'r')

        if self.type == "val":
            index = index + int(self.length * 0.7 * 0.8)
        
        if self.type == "test":
            index = index + int(self.length * 0.7)


        if index >= len(self.dataset['train']["snitch_positions"]):
            index = index - len(self.dataset['train']["snitch_positions"])
            latent_states     = self.dataset['test']["object_states"][index].astype(np.float32)
            snitch_positions  = self.dataset['test']["snitch_positions"][index]
            snitch_label      = self.dataset['test']["snitch_label"][index]
            snitch_contained  = self.dataset['test']["snitch_contained"][index]
            actions_visible   = self.dataset['test']["actions_visible"][index]
            actions_hidden    = self.dataset['test']["actions_hidden"][index]
            materials_visible = self.dataset['test']["materials_visible"][index]
            materials_hidden  = self.dataset['test']["materials_hidden"][index]
            sizes_visible     = self.dataset['test']["sizes_visible"][index]
            sizes_hidden      = self.dataset['test']["sizes_hidden"][index]
            colors_visible    = self.dataset['test']["colors_visible"][index]
            colors_hidden     = self.dataset['test']["colors_hidden"][index]

        latent_states     = self.dataset['train']["object_states"][index].astype(np.float32)
        snitch_positions  = self.dataset['train']["snitch_positions"][index]
        snitch_label      = self.dataset['train']["snitch_label"][index]
        snitch_contained  = self.dataset['train']["snitch_contained"][index]
        actions_visible   = self.dataset['train']["actions_visible"][index]
        actions_hidden    = self.dataset['train']["actions_hidden"][index]
        materials_visible = self.dataset['train']["materials_visible"][index]
        materials_hidden  = self.dataset['train']["materials_hidden"][index]
        sizes_visible     = self.dataset['train']["sizes_visible"][index]
        sizes_hidden      = self.dataset['train']["sizes_hidden"][index]
        colors_visible    = self.dataset['train']["colors_visible"][index]
        colors_hidden     = self.dataset['train']["colors_hidden"][index]
        return (
            latent_states, 
            snitch_positions,
            snitch_label,
            snitch_contained,
            actions_visible,
            actions_hidden,
            materials_visible,
            materials_hidden,
            sizes_visible,
            sizes_hidden,
            colors_visible,
            colors_hidden
        )
