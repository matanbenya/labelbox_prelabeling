# This mains load a pre-run labels from GRoundedSAM and uploads them to "covered Baby" project in labelbox, as pre labels
import os
import pandas as pd
from dataset_retriever import DatasetRetriever
from upload_prelabel import UploadPrelabel
from image_retriever import ImageLoader
from annotator import GroundedSam


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    DATAROWS_JOSN_PATH = '/Users/matanb/Downloads/export-result.ndjson'
    classes = ['Covered Baby',...]
    annotator = GroundedSam(classes = classes)

    img_loader = ImageLoader(project='ml-workspace-2', bucket_name='nanobebe_data')
    # get the dataset df
    datarows_df = DatasetRetriever(DATAROWS_JOSN_PATH).get_df()
    uploader = UploadPrelabel()

    status_df = pd.DataFrame(columns=['id', 'status'])
    for index, row in datarows_df.iterrows():
        # get the data row id
        row_id = row['id']

        # load the image
        img, image_path = img_loader.load_image(row_id=row_id)

        # generate the annotation
        annotation_path = annotator(row_id=row_id, image_path=image_path)

        # upload the pre label
        for c in classes:
            uploader(row_id=row_id, annotation_path=annotation_path, classname=c, type='segmentation')

        # update the status df and save to csv
        # status_df = status_df.append({'id': row_id, 'status': 'uploaded'}, ignore_index=True)
        # status_df.to_csv('status.csv', index=False)
