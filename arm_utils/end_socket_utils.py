import cv2

def calc_draw_dist(im0, socket, end, right=False):
    if right:
        d_l = 80 / (end[3] - end[1])  # mm/px
    else:
        d_l = 80 / (end[2] - end[0])  # mm/px
    s_h, s_v = (int(socket[0] + socket[2])) // 2, int((socket[1] + socket[3])) // 2
    if right:
        e_h, e_v = (int(end[0] + end[2])) // 2, (int(end[1] + end[3])) // 2,
    else:
        e_v = int(end[1] + end[3] + (end[3]-end[1])//3 ) // 2
        e_h = int(end[0] + end[2] + (end[2]-end[0])//2) // 2
    h_dist = (s_h - e_h)
    v_dist = (s_v - e_v)

    cv2.line(im0, (s_h, s_v), (e_h, e_v), (0, 50, 255), 2)
    cv2.line(im0, (s_h, s_v), (s_h, e_v), (0, 50, 255), 3)
    cv2.line(im0, (s_h, e_v), (e_h, e_v), (0, 50, 255), 3)
    pos = 50
    if right:
        pos=pos + im0.shape[1]//2
    cv2.putText(im0, str(h_dist * d_l), color=(0, 255, 0), fontScale=1, thickness=3,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(pos, 200))
    cv2.putText(im0, str(v_dist * d_l), color=(0, 255, 0), fontScale=1, thickness=3,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(pos, 250))
    return im0, d_l, h_dist, v_dist

def found_sockets_ends(sockets, ends, img_shape):
    if len(sockets) > 1:
                left_socket, right_socket = (sockets[0], sockets[1]) if sockets[0][0] < sockets[1][0] else (
                sockets[1], sockets[0])
    elif len(sockets) == 1:
        left_socket = sockets[0] if sockets[0][0] < img_shape[1] // 2 else None
        right_socket = sockets[0] if sockets[0][0] > img_shape[1] // 2 else None
    else:
        left_socket, right_socket = None, None
    if len(ends) > 1:
        left_end, right_end = (ends[0], ends[1]) if ends[0][0] < ends[1][0] else (ends[1], ends[0])
    elif len(ends) == 1:
        left_end = ends[0] if ends[0][0] < img_shape[1] // 2 else None
        right_end = ends[0] if ends[0][0] > img_shape[1] // 2 else None
    else:
        left_end, right_end = None, None
    return left_socket, right_socket, left_end, right_end
