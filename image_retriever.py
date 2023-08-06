from google.cloud import storage
import cv2
import numpy as np

class ImageLoader:
    def __init__(self, project='ml-workspace-2', bucket_name='nanobebe_data'):
        self.client = storage.Client(project=project)
        self.bucket = self.client.get_bucket(bucket_name)


    def __call__(self, url):
        path = url.split('nanobebe_data/')[1].split('?')[0]
        blob = self.bucket.blob(path)
        # download the file
        file = blob.download_as_string()
        # read as jpeg
        img = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_COLOR)
        return img, path



