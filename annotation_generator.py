from autodistill_grounded_sam import GroundedSAM
from autodistill_yolov8 import YOLOv8
from autodistill.detection import CaptionOntology
import cv2
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import numpy as np

base_model = GroundedSAM(ontology=CaptionOntology({"blanket": "blanket", "baby": "baby"}))

img = cv2.imread(r'/home/matanb/Downloads/index.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
res = base_model.predict(r'/home/matanb/Downloads/index.jpeg')

img_masked = img.copy()
for mask in res.mask:
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(255*mask, cv2.COLOR_GRAY2RGB)
    img_masked = cv2.addWeighted(img_masked, 0.5, mask, 0.5, 0)
plt.imshow(img_masked)
plt.show()

import time

t = time.time()
res = base_model.predict(r'/home/matanb/Downloads/index.jpeg')
print(time.time()-t)


base_model.label("./context_images", extension=".jpeg")

class Annotator:
    def __init__(self, classes = {"blanket": "blanket", "baby": "baby"}):
        base_model = GroundedSAM(ontology=CaptionOntology(classes))
    def annotate(self, img_url, output_path = output_path):
