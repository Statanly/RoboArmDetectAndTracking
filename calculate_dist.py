import cv2

def calc_draw_dist(im0, socket, end):
    d_l = 100 / (socket[3] - socket[1])  # mm/px
    s_h, s_v = (int(socket[0] + socket[2])) // 2, int((socket[1] + socket[3])) // 2
    e_h, e_v = (int(end[0] + end[2])) // 2, (int(end[1] + end[3])) // 2,
    h_dist = (s_h - e_h)
    v_dist = (s_v - e_v)

    cv2.line(im0, (s_h, s_v), (e_h, e_v), (0, 50, 255), 2)
    cv2.line(im0, (s_h, s_v), (s_h, e_v), (0, 50, 255), 3)
    cv2.line(im0, (s_h, e_v), (e_h, e_v), (0, 50, 255), 3)
    cv2.putText(im0, str(h_dist * d_l), color=(0, 255, 0), fontScale=1.5, thickness=3,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(50, 200))
    cv2.putText(im0, str(v_dist * d_l), color=(0, 255, 0), fontScale=1.5, thickness=3,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                org=(50, 250))
    return im0, d_l, h_dist, v_dist