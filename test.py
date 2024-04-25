# import cv2
# from ultralytics import YOLO
# from utils.img_utils import draw_fall_img
# import numpy as np
# cap = cv2.VideoCapture(0)
# model = YOLO('weights/human_detection.pt')
import cv2

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#     results = model.predict(frame)
#     frame = draw_fall_img(results)
#     cv2.imshow('frame', frame)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break


# cap.release()
# cv2.destroyAllWindows()
# results = model.predict('./src/bus.jpg')
# print(results[0].boxes)
# print(results[0].boxes.conf[0])
# frame = draw_fall_img(results)
# cv2.imshow('frame', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

from utils.img_utils import img_rotate, draw_fall_img, rotate_points
from ultralytics import YOLO

ori_img = cv2.imread('src/fall1.jpg')
# cv2.imshow('img0', img)
# print(ori_img.shape)
img = img_rotate(ori_img, angle=270)
# print(img.shape)
# cv2.imshow('img1', img)
model = YOLO('weights/human_detection.pt')
result = model.predict(img)
print(result[0].boxes[0].xywh[0].tolist()[2:])
# print(result[0].boxes)
# boxes = result[0].boxes.xyxy[0].tolist()
# box = [[boxes[0], boxes[1]], [boxes[2], boxes[3]]]
# detect_img = draw_fall_img(box, img)
# cv2.imshow('img', detect_img)
# new_box = []
# new_box.append(rotate_points((boxes[0], boxes[1]), angle=360-270, center=(img.shape[1]//2, img.shape[0]//2)))
# new_box.append(rotate_points((boxes[2], boxes[3]), angle=360-270, center=(img.shape[1]//2, img.shape[0]//2)))
# img = img_rotate(img, angle=360-270)
# # print(img.shape)
# detect_img1 = draw_fall_img(new_box, ori_img)
#
# cv2.imshow('img1', detect_img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# b526948590e19e74f18bfcd32c0c5ce5b68bc69b
# 6504cbc951860b74e9fb80debb54095af04bfc0b
