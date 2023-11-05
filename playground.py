import labelbox as lb
import labelbox.data.annotation_types as lb_types
import uuid
import numpy as np
import os
import json
import cv2


API_KEY =r'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGpvYnBlY2kwNHk0MDd5cTViNGIzMzAwIiwib3JnYW5pemF0aW9uSWQiOiJjbGQxdjY4cmMwZHV0MDcyNzg5bHgydGkxIiwiYXBpS2V5SWQiOiJjbGtpYWlrZXoxY2hsMDczZmdvemwyZ2JoIiwic2VjcmV0IjoiYzFjMWFhNDY0YjAxYWY4M2I2MDI1Y2JmZGUzZTM1MWQiLCJpYXQiOjE2OTAyODkxNDYsImV4cCI6MjMyMTQ0MTE0Nn0.fPe3UwQl3lvruxoE7BBw461o050S8zlJ0V1kvr3824I'
client = lb.Client(API_KEY)




# get the Covered Baby project
project = client.get_project('cljvb348h00ef07xf2t4u3fah')
# get the dataset


data_row = client.get_data_row(r'cld36xf2v5480071p5qlf1vid')
# assign global key
global_key = str(uuid.uuid4())
global_key_data_row_inputs = [
  {"data_row_id": data_row.uid, "global_key": global_key}
]
client.assign_global_keys_to_data_rows(global_key_data_row_inputs)

# get data row metadata
height = data_row.media_attributes['height']
width = data_row.media_attributes['width']
mask = np.zeros((height, width), dtype=np.uint8)
# enter a circle



# Identifying what values in the numpy array correspond to the mask annotation
color = (255, 255, 255)
mask_data = lb_types.MaskData(file_path="blanket0.png")

# Python annotation
mask_annotation = lb_types.ObjectAnnotation(
  name = "Blanket", # must match your ontology feature"s name
  value=lb_types.Mask(mask=mask_data, color=color),
)

# Python annotation
bbox_annotation = lb_types.ObjectAnnotation(
    name="Covered Baby",  # must match your ontology feature"s name
    value=lb_types.Rectangle(
        start=lb_types.Point(x=350, y=200),  #  x = left, y = top
        end=lb_types.Point(x=500, y=450),  # x= left + width , y = top + height
    ))



labels = []
annotations = [
    mask_annotation
]
labels.append(
    lb_types.Label(data=lb_types.ImageData(global_key=global_key),
                   annotations=annotations))



# Upload MAL label for this data row in project
upload_job = lb.MALPredictionImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="mal_job"+str(uuid.uuid4()),
    predictions=labels)

print(f"Errors: {upload_job.errors}", )
print(f"Status of uploads: {upload_job.statuses}")

# Upload label for this data row in project  - GROund TRUTJ
upload_job = lb.LabelImport.create_from_objects(
    client = client,
    project_id = project.uid,
    name="label_import_job"+str(uuid.uuid4()),
    labels=labels)

print(f"Errors: {upload_job.errors}", )
print(f"Status of uploads: {upload_job.statuses}")




# ================= Auto distill
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
img = cv2.imread(r'/Users/matanb/Desktop/Screenshot 2023-05-04 at 9.21.15.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2R)
res = base_model.predict(r'/Users/matanb/Desktop/Screenshot 2023-05-04 at 9.21.15.png')

img_masked = img.copy()
for mask in res.mask:
    mask = mask.astype(np.uint8)
    mask = cv2.cvtColor(255*mask, cv2.COLOR_GRAY2RGB)
    img_masked = cv2.addWeighted(img_masked, 0.5, mask, 0.5, 0)
plt.imshow(img_masked)
plt.show()
