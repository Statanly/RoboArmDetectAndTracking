"""
Dataloader for get two video from cameras|video files
"""
import os
from pathlib import Path


import numpy as np

from PIL import ExifTags, Image


import cv2

#https://github.com/opencv/opencv/issues/17734

# Parameters
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html


print(cv2.getBuildInformation())
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path_front, path_side, img_size=640, stride=32, auto=True):
        p_front = str(path_front)  # os-agnostic absolute pa
        is_url_front = p_front.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if not is_url_front:
            p_front = str(Path(path_front).resolve())
        if os.path.isfile(p_front) or is_url_front:
            files_front = [p_front]  # files
        else:
            raise Exception(f'ERROR: {p_front}  does not exist')

        p_side = str(path_side)
        is_url_side = p_side.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if not is_url_side:
            p_side = str(Path(path_side).resolve())  # os-agnostic absolute path
        if os.path.isfile(p_side) or is_url_side:
            files_side = [p_side]  # files
        else:
            raise Exception(f'ERROR: {p_side}  does not exist')


        images_side = [x for x in files_side if x.split('.')[-1].lower() in IMG_FORMATS]
        videos_side = [x for x in files_side if x.split('.')[-1].lower() in VID_FORMATS or (x.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')))]
        ni_s, nv_s = len(images_side), len(videos_side)

        images_front  = [x for x in files_front if (x.split('.')[-1].lower() in IMG_FORMATS) ]
        videos_front  = [x for x in files_front if x.split('.')[-1].lower() in VID_FORMATS or (x.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')))]
        ni_f, nv_f = len(images_front), len(videos_front)


        ni = min(ni_s, ni_f)
        nv = min(nv_s, nv_f)

        self.img_size = img_size
        self.stride = stride
        self.files = [(f, s) for f, s in zip(files_front, files_side)]
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if any(videos_front) and any(videos_side):
            self.new_video(videos_front[0] , videos_side[0]) # new video
        else:
            self.cap_front = None

        assert self.nf > 0, f'No images or videos found in {p_front} or {p_side}. ' \
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
            try:
                ret_val_f, img_f0 = self.cap_front.read()
                ret_val_s, img_s0 = self.cap_side.read()

                # cv2.imshow('front', cv2.resize(img_f0, (img_f0.shape[1]//2, img_s0.shape[0]//2)))
                # cv2.imshow('side', cv2.resize(img_s0,  (img_s0.shape[1]//2, img_s0.shape[0]//2)))
            except AttributeError:
                print("No frame")
            while ret_val_f is None or ret_val_f is None:
                ret_val_f, img_f0 = self.cap_front.read()
                ret_val_s, img_s0 = self.cap_side.read()
            if not ret_val_s or not ret_val_f:
                # print(img_s0, img_f0)
                # print("No next frame"
                #       "")
                self.cap_front.release()
                self.cap_side.release()
                raise StopIteration
            else:
                # print(img_s0, img_f0)
                w, h = min(img_f0.shape[1], img_s0.shape[1]), min(img_f0.shape[0], img_s0.shape[0])
                img_f0 = cv2.resize(img_f0, (w, h))
                img_s0 = cv2.resize(img_s0, (w, h))

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
        # img0 = self.high_contrast(img0)
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride, auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap_front, self.cap_side, s

    def high_contrast(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)

        # Applying CLAHE to L-channel
        # feel free to try different values for the limit and grid size:
        # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(8, 8))
        # cl = self.clahe.apply(l_channel)
        cl = cv2.equalizeHist(l_channel)
        # merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))

        # Converting image from LAB Color model to BGR color spcae
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        return enhanced_img

    def new_video(self, path_front, path_side):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
        self.frame = 0
        try:
            self.cap_front = cv2.VideoCapture(path_front)
            self.frames_front = max(int(self.cap_front.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.cap_side = cv2.VideoCapture(path_side)
            self.frames_side = max(int(self.cap_side.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
        except Exception as e:
            print("No frame", e)

        # for _ in range(30):
        #     _, _ = self.cap_front.read()

    def __len__(self):
        return self.nf  # number of files




