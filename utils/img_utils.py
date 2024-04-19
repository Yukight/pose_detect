import cv2
import os
from PIL import Image
import numpy as np
import math
def draw_fall_img(box, img):
    try:
        # for i in range(len(results[0].boxes.xyxy)):
        # info = results[0].boxes.xyxy[0].tolist()
        x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        detect_img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        return detect_img
    except:
        return img


def img_rotate(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    rotated_img = cv2.warpAffine(img, M, (new_w, new_h))

    return rotated_img



def rotate_points(point, center, angle):
    x, y = point
    x_c, y_c = center
    x_rel, y_rel = x - x_c, y - y_c
    angle_rad = np.deg2rad(angle)
    x_new = x_rel * np.cos(angle_rad) - y_rel * np.sin(angle_rad)
    y_new = x_rel * np.sin(angle_rad) + y_rel * np.cos(angle_rad)
    x_new += x_c
    y_new += y_c
    return (x_new, y_new)


def get_detect_info(log_w, log_h):
    try:
        # x1, y1, x2, y2 = box[0][0], box[0][1], box[1][0], box[1][1]
        # print([x1, y1, x2, y2])
        # status_info = (round((x2 - x1) / (y2 - y1), 1))
        status_info = (round((log_w / log_h), 1))
        print(status_info)
        if status_info > 1.5:
            return 'lie_down'
        if status_info <= 1.5 and status_info > 0.7:
            return 'half_squat'
        else:
            return 'stand'
    except:
        return 'half_squat'

# def log_img(log_img_folder, status_message, frame_count, processed_frame):
#     img_path = os.path.join(log_img_folder, f"{status_message}_{frame_count}.jpg")
#     cv2.imwrite(img_path, processed_frame)
#
# def delete_all_files_in_directory(directory_path):
#     if not os.path.exists(directory_path):
#         print(f" {directory_path} does not exists")
#         return
#
#     entries = os.listdir(directory_path)
#
#     for entry in entries:
#         full_path = os.path.join(directory_path, entry)
#         if os.path.isfile(full_path) or os.path.islink(full_path):
#             os.remove(full_path)
#         else:
#             print(f"{full_path} passed")
#     print(f" {directory_path} have already cleared")
