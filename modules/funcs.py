import cv2
import matplotlib.pyplot as plt
from modules.utils import load_class_names, detect_objects, num_of_person
from modules.darknet import Darknet
cfg_file = './modules/cfg/yolov3.cfg'
weight_file = './modules/weights/yolov3.weights'
namesfile = './modules/data/coco.names'

nms_thresh = 0.6
iou_thresh = 0.4
plt.rcParams['figure.figsize'] = [24.0, 14.0]

# Load the image
webcam = cv2.VideoCapture(0)


class Rec:
    def __init__(self) -> None:
        self.m = Darknet(cfg_file)
        self.m.load_weights(weight_file)
        self.class_names = load_class_names(namesfile)

    def get_persons(self) -> int:
        (_, img) = webcam.read()
        # Convert the image to RGB
        original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image,
                                   (self.m.width, self.m.height))
        boxes = detect_objects(self.m, resized_image, iou_thresh, nms_thresh)
        return num_of_person(boxes, self.class_names)
