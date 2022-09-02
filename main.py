import math
import time
import os

# limit the number of cpus used by high performance libraries


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

from arm_utils.parse_args import parse_opt
from arm_utils.end_socket_utils import calc_draw_dist, found_sockets_ends

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'Rooky' / 'python') not in sys.path:
    sys.path.append(str(ROOT / 'Rooky' / 'python'))  # add Rooky ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging

from yolov5.models.common import DetectMultiBackend
# from yolov5.utils.dataloaders import VID_FORMATS, LoadImages
from arm_utils.dataloaders import VID_FORMATS, LoadImages
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT
from yolov5.utils.general import print_args

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


def check_path(source):
    is_file = Path(source).suffix[1:] in (VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    return is_file, is_url


arms_joints_dgs = {
    'left_arm_1_joint': 0.0,
    'left_arm_2_joint': 0.0,
    'left_arm_3_joint': 0.0,
    'left_arm_4_joint': 0.0,
    'left_arm_5_joint': 0.0,
    'left_arm_6_joint': 0.0,
    'left_arm_7_joint': 0.0,
}
ros = True
try:
    # lib to working with robo arm
    # Импортируем необходимые библиотеки
    # библиотека работы с ROS
    import rospy

    # Данный тип сообщений необходим для trajectory_msgs
    from std_msgs.msg import *

    # Сообщения для описания траектории движения
    from trajectory_msgs.msg import *

    from arm_utils.move_joints import ControlJoints
except ImportError as e:
    LOGGER.warning("No ros lib found.")
    ros = False
try:
    from Rooky.python import Rooky2
except:
    LOGGER.warning("No Rooky found")


@torch.no_grad()
def run(
        source_front='0',
        source_side='0',
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_conf=False,  # save confidences in --save-txt labels
        save_vid=False,  # save confidences in --save-txt labels
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        not_move_arm=False,
        connection_time = 30,
        delay = 5
):
    if not not_move_arm:
        rospy.init_node('joint_control_sim_test')
        # Создадим узел ROS
        node = ControlJoints("left")

            # node.reset_joints()

    flag_can_move = True
    flag_arm_down = True
    flag_arm_side = True
    x, y = 0, 0
    delay = int(delay)


    flag_to_disconnect = False
    time_to_disconnect = -1
    time_end = -1
    time_no_arm = 0

    source_front = str(source_front)
    source_side = str(source_side)

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader

    show_vid = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference


    nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources
    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources

    dataset = LoadImages(source_front, source_side, img_size=imgsz, stride=stride, auto=pt)
    time_to_connect = time.time() + int(connection_time) + 3
    print(time.time(), time_to_connect)
    for frame_idx, (path, im, im0s, vid_cap_f, vid_cap_s, s) in enumerate(dataset):
        key=cv2.waitKey(1) &0xFF
        if key == ord("q"):
            break
        if frame_idx%2 or frame_idx%3:
            continue
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                   max_det=max_det)
        dt[2] += time_sync() - t3

        # Process detections
        if len(pred) > 4:
            raise ValueError('Something went wrong. Detected more than 2 sockets and 2 ends')
        ends = []
        sockets = []
        for i, det in enumerate(pred):  # detections per image
            seen += 1

            p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            p = Path(p[0])
            if source_front.endswith(VID_FORMATS):
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # folder with imgs
            else:
                save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            s += '%gx%g ' % im.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += str(n)
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:

                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        if cls == 0:
                            ends.append(bboxes)
                        else:
                            sockets.append(bboxes)
                        if save_vid or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                                                  (
                                                                      f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                #print(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)')

            else:
                strongsort_list[i].increment_ages()
                LOGGER.info('No detections')

        '''
        Found distances between sockets and ends.
        '''
        left_socket, right_socket, left_end, right_end = found_sockets_ends(sockets=sockets, ends=ends,
                                                                            img_shape=im0.shape)

        h_dist_right, v_dist_right, h_dist_left, v_dist_left = None, None, None, None
        d_l, d_r = 0, 0
        if left_socket is not None and left_end is not None:
            im0, d_l, h_dist_left, v_dist_left = calc_draw_dist(im0, left_socket, left_end)

        if right_socket is not None and right_end is not None:
            im0, d_r, h_dist_right, v_dist_right = calc_draw_dist(im0, right_socket, right_end, right=True)
        L1 = 256.1
        L2 = 249.1+89+85+35

        if not not_move_arm and flag_can_move and not flag_to_disconnect:
            time_end = time.time()
            '''
            Case 1: No arm found, so we up the arm to start position - shoulder to 90, elbow to 60. 
            '''
            if len(ends) == 0:
                print('no arm', time_no_arm)
                node._positions = [0.7, 0.0, 0.0, 1.3, 0.0, -0.13, 0.0]

                if flag_can_move:
                    time_end = time_end + delay
                flag_can_move = False
                flag_arm_down = True
                node.move_all_joints(1.0)
                time_no_arm += 1

            #Case 2: Using inverse kinematics

            elif flag_arm_down and v_dist_right and h_dist_right and h_dist_left: #no moving before
                '''
                                Finding x, y using forward kinematics.
                                x = XA + x' = L1*cos(Q1) + L2*cos(Q1+Q2)
                                y = YA + y' = L1*sin(Q1) + L2*sin(Q1+Q2)
                                '''
                try:
                    x = L1 * math.cos(node._positions[0]) + L2 * math.cos((node._positions[0] + node._positions[3]))
                    y = L1 * math.sin(node._positions[0]) + L2 * math.sin((node._positions[0] + node._positions[3]))
                except:
                    print("Error in locating arms end.")
                    break
                print("First move processing...")
                dx, dy, ds = 0, 0, 0
                if v_dist_right and abs(v_dist_right * d_r) > 5:
                    if flag_can_move:
                        time_end = time_end + delay

                    dx = v_dist_right*d_r

                    time_no_arm = 0
                    flag_can_move = False

                if h_dist_right and abs(h_dist_right * d_r) > 50:
                    if flag_can_move:
                        time_end = time_end + delay
                    ## 55 - compensation of width of reciever and socket, 5 - to have distance.
                    dy = h_dist_right * d_r-70

                    time_no_arm = 0
                    flag_can_move = False

                if h_dist_left and abs(h_dist_left * d_l) > 10:
                    if flag_can_move:
                        time_end = time_end + delay

                    ds = h_dist_left * d_l

                    time_no_arm = 0
                    flag_can_move = False

                '''Find angles:
                Q1 = q1 - q2 = arccos( x/B ) - arccos(L1^2 - L2^2 + B^2 / 2*B*L1 )
                Q2 = PI - arccos( L1^2 + L2^2 - B^2 / 2*L1*L2  )
                '''
                print("dx", "dy", str(dx), str(dy))
                try:
                    x, y = x+dx, y+dy
                    # print("X, y new", str(x), str(y))
                    B = math.sqrt(x*x+y*y)
                    v = (L1**2 - L2**2 + B**2) / (2*B*L1)
                    q1 = math.acos(x/B) - math.acos(v)
                    v = (L1**2 + L2**2 - B**2) / (2*L1*L2)
                    q2 = (math.pi - math.acos(v))
                    q3 = -math.atan(ds/y)*1.4
                    if q3 < 0:
                        print("Error. Please move arm more to the right.")
                        break;
                    node._positions[1] = q3
                    node._positions[6] = -q3*0.1
                    node.move_all_joints(1)
                    rospy.sleep(1)
                    node._positions[0] = q1
                    node._positions[3] = q2
                    print("angles", q1, q2, q3)
                except ValueError as e:
                    node.reset_joints()
                    print("Error in detecting new position. Mostly that mean that arm is located too far. "
                          "Please move it closer. ", str(v))

                # print('move all',time.time(), time_end, node._positions)
                flag_arm_down = False
                flag_can_move = False
                node.move_all_joints(1.5)
            elif  v_dist_right and h_dist_right and h_dist_left:
                if h_dist_left and abs(h_dist_left * d_l) > 3:
                    if flag_can_move:
                        time_end = time_end + delay

                    if abs(h_dist_left*d_l) > 50:
                        d = 0.1
                    else:
                        d = 0.02

                    if h_dist_left > 0:
                        node._positions[1] = node._positions[1] - d
                        node._positions[5] = node._positions[5] - 0.005
                        node._positions[6] = node._positions[6] - 0.005
                    else:
                        node._positions[1] = node._positions[1] + d
                        node._positions[6] = node._positions[6] + 0.005
                    flag_can_move = False
                if h_dist_right:
                    # move arm little closer
                    if abs(h_dist_right * d_r) > 38 and abs(v_dist_right) < 40:
                        if flag_can_move:
                            time_end = time_end + delay

                        if node._positions[3] < 0.15:
                            print(node._positions)
                            print("Arm is located too far. Please move it closer.")
                            break
                        if h_dist_right * d_r > 75:
                            node._positions[0] += 0.15
                            node._positions[3] -= 0.25
                            if node._positions[1]>0.1:
                                node._positions[1] -= 0.05

                        elif h_dist_right * d_r > 30:
                            node._positions[0] += 0.05
                            node._positions[3] -= 0.1
                            if node._positions[1]>0.1:
                                node._positions[1] -= 0.02
                        flag_can_move = False
                if v_dist_right:
                    # move arm little closer
                    if abs(v_dist_right * d_r) > 2:
                        if flag_can_move:
                            time_end = time_end + delay
                        if v_dist_right * d_r > 0:
                            node._positions[0] -= 0.03
                        else:
                            node._positions[0] += 0.03
                        flag_can_move = False

                print('Move all with small moves.', time.time(), time_end, node._positions)
                node.move_all_joints(1.0)
            # elif (h_dist_left and abs(h_dist_left * d_l) > 100 or h_dist_right and abs(
            #         h_dist_right * d_r) > 100 or v_dist_right and abs(v_dist_right * d_r) > 40) and flag_can_move:
            #     print("Arm was moving strange...")
            #     flag_arm_down = True

            else:
                print("Try to find arm...")


        if h_dist_left and h_dist_right and v_dist_right:
            if abs(h_dist_left*d_l)<3 and abs(h_dist_right)*d_r<40 and abs(v_dist_right*d_r)<5 and not flag_to_disconnect:
                print("Seems like connected. Waiting...")
                flag_can_move = False
                flag_to_disconnect = True
                time_to_disconnect = time.time()+3
                time_end = time.time()+100

            # Stream results
        im0 = annotator.result()
        if show_vid:
            im0 = cv2.resize(im0, (im0.shape[1] // 3, im0.shape[0] // 3))
            cv2.imshow(str(p), im0)
            # cv2.waitKey(1)  # 1 millisecond


        # Save results (image with detections)
        if save_vid:
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap_f:  # video
                    fps = vid_cap_f.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap_f.get(cv2.CAP_PROP_FRAME_WIDTH)) * 2
                    h = int(vid_cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            LOGGER.info(im0.shape)
            vid_writer[i].write(im0)
        if not not_move_arm:
            if flag_to_disconnect:
                if \
                        time.time() > time_to_disconnect:
                    print("Disconnectd after connection")
                    node._positions[3] = 1.3
                    node.move_all_joints(1.5)
                    rospy.sleep(2)
                    node._positions[0] -= 0.3
                    node.move_all_joints(1.5)
                    rospy.sleep(2)
                    node.reset_joints()
                    break

            if time.time() > time_end:
                flag_can_move = True
            if time_no_arm > 5:
                print(time_no_arm)
                node.reset_joints()

                break
            if time.time() > time_to_connect:
                print("Cannot connect in time. Please try again.")
                node.reset_joints()
                break
        prev_frames[i] = curr_frames[i]

    if not not_move_arm:
        node._positions[3] -=0.2
        node.move_all_joints(1.5)
        rospy.sleep(0.5)
        node.reset_joints()
    try:
        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
        if save_vid:
            s = ''
            LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)
    except ZeroDivisionError:
        pass

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt(ROOT, WEIGHTS)
    # print_args(opt)

    main(opt)