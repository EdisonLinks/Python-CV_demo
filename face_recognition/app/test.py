import cv2
import numpy as np
from detection.face_detect import MTCNN
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

img = cv2.imread("./dataset/test.jpg")

# img = cv2.imdecode(np.fromfile("./dataset/test.jpg", dtype=np.uint8), -1)

mtcnn = MTCNN(model_path="save_model/mtcnn")
imgs, boxes_c = mtcnn.infer_image(img)

print(imgs)
print(boxes_c)