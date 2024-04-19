from ultralytics import YOLO
import os
import cv2

from utils.img_utils import draw_fall_img, get_detect_info, img_rotate, rotate_points


class Fall_Detect:
    def __init__(self, weights_path, sensitivity):
        self.weights_path = weights_path
        self.model = YOLO(self.weights_path)
        self.sensitivity = sensitivity
        self.rotate_angle = [0, 90, 180, 270]

    def fall_predict(self, img):
        results = 0
        max_results = 0
        for angle in self.rotate_angle:
            rotated_img = img_rotate(img, angle)
            result = self.model.predict(rotated_img)  # predict on an image
            try:
                results = result[0].boxes.conf[0]
            except:
                results = 0
            if results > max_results:
                max_results = results
                box_info = result[0].boxes.xyxy[0].tolist()
                box = [[box_info[0], box_info[1]], [box_info[2], box_info[3]]]
                detect_img = draw_fall_img(box, rotated_img)
                log_angle = angle
                if angle == 90 or angle == 270:
                    log_w, log_h = result[0].boxes[0].xywh[0].tolist()[3], result[0].boxes[0].xywh[0].tolist()[2]
                elif angle == 0 or angle == 180:
                    log_w, log_h = result[0].boxes[0].xywh[0].tolist()[2], result[0].boxes[0].xywh[0].tolist()[3]
                # box_info = rotate_points(box_info, 360-angle, rotated_img.shape[0], rotated_img.shape[1])


        if max_results > self.sensitivity:
            # new_box = []
            # new_box.append(
            #     rotate_points((box[0][0], box[0][1]), angle=360 - log_angle, center=(detect_img.shape[1] // 2, detect_img.shape[0] // 2)))
            # new_box.append(
            #     rotate_points((box[1][0], box[1][1]), angle=360 - log_angle, center=(detect_img.shape[1] // 2, detect_img.shape[0] // 2)))
            detect_info = get_detect_info(log_w, log_h)
            detect_img = img_rotate(detect_img, 360 - log_angle)
            # model_pose = YOLO('../weights/pose.pt')
            # model_pose.predict(img, show=True, save=True)[0].show()
            return detect_info, detect_img
        else:
            return None, img
# Predict with the model

# if __name__ == '__main__':
#     test = Fall_Detect()
#     img_path = 'D:/self_dir/pose_detect/src/block.jpg'
#     test.fall_predict(img_path)
