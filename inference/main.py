from config import Config
from utils.detector_num import Detector as Detector_num
import cv2
import glob
import time

config = Config()
confidence = 0.15
num_classes = 2
conf_str = str(confidence).replace(".","_")
detectorObj_num = Detector_num()
detectorObj_num.setup("own_detector", config.PATH_TO_CKPT, config.PATH_TO_LABELS, num_classes, confidence)


print("detector loaded")

folder = "images/"

ctr = 0
tot_time = 0
for filename in glob.glob(folder + "/*.png"):
    print(filename)
    image = cv2.imread(filename)
    
    boundingBoxImage, filtered_detections, filtered_boxes , time_taken = detectorObj_num.getInferenceResults(image.copy())
    print(f"Time taken is {time_taken}")
    tot_time += time_taken

    cv2.imwrite("out22/" + str(ctr) + ".jpg", boundingBoxImage)
    ctr += 1
    print("done")

avg_time = tot_time/ctr
print(f"Tot time is {tot_time}")
print(f"Avg time is {avg_time}")
print(f"Avg fps is {1/avg_time}")

