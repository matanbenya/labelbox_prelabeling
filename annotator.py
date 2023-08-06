import cv2
import numpy as np
import supervision as sv
from typing import List
import torch
import torchvision
import os
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt

# # first mk weights dir
# os.system(r'mkdir -p /GroundedSam')
# os.system(r'!wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
# download sam weights
# os.system(r'!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

path = r'GroundedSam/GroundingDINO_SwinT_OGC.py'
GROUNDING_DINO_CHECKPOINT_PATH = r'/Users/matanb/Downloads/groundingdino_swint_ogc.pth'
# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = r'Users/matanb/Downloads/sam_vit_h_4b8939.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
grounding_dino_model = Model(
    model_config_path=r'GroundedSam/GroundingDINO_SwinT_OGC.py',
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device=DEVICE)
# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam_predictor = SamPredictor(sam)

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

class GroundedSam:
    def __init__(self, sam_predictor: SamPredictor, grounding_dino_model: Model, classes: List[str]):
        self.sam_predictor = sam_predictor
        self.grounding_dino_model = grounding_dino_model
        self.classes = classes

    def predict(self, image: np.ndarray, box_threshold: float, text_threshold: float,
                nms_threshold: float) -> np.ndarray:
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=self.classes,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        # replace None with 0
        detections.class_id[detections.class_id == None] = 0

        nms = True
        if nms:
            # NMS post process
            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                nms_threshold
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        return detections

    def predict_with_classes(self, image: np.ndarray, box_threshold: float, text_threshold: float,
                             nms_threshold: float) -> np.ndarray:
        detections = self.predict(image=image, box_threshold=box_threshold, text_threshold=text_threshold,
                                  nms_threshold=nms_threshold)
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{self.classes[class_id]} {confidence:0.2f}"
            for _, _, confidence, class_id, _
            in detections]
        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        return annotated_image

    def save(self, img_path, detections):
        # for each class, create a mask and save as a png  with the file name as the class name
        for class_id in np.unique(detections.class_id):
            # create a mask of the class
            mask = detections.mask[detections.class_id == class_id].sum(axis=0)
            # the image should be uint8 0-255
            mask = (mask * 255).astype(np.uint8)
            # save the mask
            cv2.imwrite(os.path.join(img_path, self.classes[class_id] + ".png"), mask)
        return True


