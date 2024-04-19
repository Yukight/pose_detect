import cv2
import os
import random
from core.core import Fall_Detect
# from utils.img_utils import delete_all_files_in_directory, log_img
from ultralytics import YOLO

class Detector:

    def __init__(self):
        self.log_img_folder = "./src/log_img"
        self.fall_detect = Fall_Detect(weights_path='../weights/human_detection.pt', sensitivity=0.8)
        self.log_info = dict()
        self.gap = 5
        self.max_restore = 50
        self.frame_count = 0
        self.model_pose = YOLO('../weights/pose.pt')
        self.L1_distance = 50
    def max_restore_judge(self):
        if self.frame_count > self.max_restore:
            # oldest_img_path = os.path.join(self.log_img_folder, f"{status_message}_{self.frame_count - 50}.jpg")
            # try:
            #     os.remove(oldest_img_path)
            # except:
            #     pass
            del self.log_info[f'{self.frame_count - self.max_restore}']

    def status_judge(self, status_message):
        print(status_message)
        if self.frame_count > self.gap:
            if status_message == 'lie_down' and self.log_info[f'{self.frame_count - self.gap}'] == 'stand':
                print("Someone's fallen")
                return True
            else:
                return False
    def run(self):
        cap = cv2.VideoCapture(0)
        # delete_all_files_in_directory(self.log_img_folder)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            # rand_num = random.randint(1, 100)
            # if rand_num % 10 == 0:
            #     frame = cv2.imread('./src/fall2.jpg')
            # else:
            #     frame = cv2.imread('./src/block.jpg')
                
            status_message, detect_img, box = self.fall_detect.fall_predict(frame)

            self.frame_count += 1
            
            # log_img(self.log_img_folder, status_message, self.frame_count, processed_frame)

            self.log_info[f'{self.frame_count}'] = status_message

            self.max_restore_judge()

            print(self.frame_count)

            if(self.status_judge(status_message)):
                # model_pose.predict(frame, show=True, save=True)[0].show()
                cv2.imshow('detect_img', detect_img)
                cv2.waitKey(0)
                Results = self.model_pose.predict(frame, save=True)
                try:
                    l_x, l_y, r_x, r_y = Results[0].boxes.xyxy[0].tolist()
                except:
                    continue
                dis = abs(l_x - box[0][0]) + abs(l_y - box[0][1]) + abs(r_x - box[1][0]) + abs(r_y - box[1][1])
                if dis < self.L1_distance:
                    self.log_info.clear()
                    self.frame_count = 0
                    break

            cv2.imshow('frame', detect_img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test = Detector()
    test.run()
