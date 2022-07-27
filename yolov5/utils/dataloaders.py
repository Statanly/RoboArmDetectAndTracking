# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    except Exception:
        pass

    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
            8: Image.ROTATE_90,}.get(orientation)
        if method is not None:
            image = image.transpose(method)
            del exif[0x0112]
            image.info["exif"] = exif.tobytes()
    return image


class InfiniteDataLoader(dataloader.DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path_front, path_side, img_size=640, stride=32, auto=True):
        p_front = str(Path(path_front).resolve())  # os-agnostic absolute path
        if os.path.isfile(p_front) and os.path.isfile(p_front):
            files_front = [p_front]  # files
        else:
            raise Exception(f'ERROR: {p_front}  does not exist')

        p_side = str(Path(path_side).resolve())  # os-agnostic absolute path
        if os.path.isfile(p_side) and os.path.isfile(p_front):
            files_side = [p_side]  # files
        else:
            raise Exception(f'ERROR: {p_side}  does not exist')

        images_front  = [x for x in files_front if x.split('.')[-1].lower() in IMG_FORMATS]
        videos_front  = [x for x in files_front if x.split('.')[-1].lower() in VID_FORMATS]
        ni_f, nv_f = len(images_front), len(videos_front)

        images_side = [x for x in files_side if x.split('.')[-1].lower() in IMG_FORMATS]
        videos_side = [x for x in files_side if x.split('.')[-1].lower() in VID_FORMATS]
        ni_s, nv_s = len(images_side), len(videos_side)
        ni = min(ni_s, ni_f)
        nv = min(nv_s, nv_f)

        self.img_size = img_size
        LOGGER.info(f' img size {img_size}')
        self.stride = stride
        self.files = [(f, s) for f, s in zip(files_front, files_side)]
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        if any(videos_front) and any(videos_side):
            self.new_video(videos_front[0] , videos_side[0]) # new video
        else:
            self.cap_front = None

        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val_f, img_f0 = self.cap_front.read()
            ret_val_s, img_s0 = self.cap_side.read()
            if not ret_val_s or not ret_val_f:
                self.cap_front.release()
                self.cap_side.release()
                raise StopIteration

            self.frame += 1
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames_front}) {path}: '

        else:
            # Read image
            self.count += 1
            img_f0 = cv2.imread(path)  # BGR
            img_s0 = cv2.imread(path)
            assert img_f0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}: '
        img0 = np.concatenate((img_f0, img_s0), axis=1)
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap_front, self.cap_side, s

    def new_video(self, path_front, path_side):
        self.frame = 0
        self.cap_front = cv2.VideoCapture(path_front)
        self.frames_front = int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cap_side = cv2.VideoCapture(path_side)
        self.frames_side = int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


# Ancillary functions --------------------------------------------------------------------------------------------------
def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder


def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(str(path) + '_flat')
    create_folder(new_path)
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)



