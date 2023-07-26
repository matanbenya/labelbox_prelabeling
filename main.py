# This mains load a pre-run labels from GRoundedSAM and uploads them to "covered Baby" project in labelbox, as pre labels
import os
import pandas as pd
from dataset_retriever import DatasetRetriever
from upload_prelabel import UploadPrelabel




# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    DATAROWS_JOSN_PATH = ''
    classes = ['Covered Baby',...]

    # get the dataset df
    datarows_df = DatasetRetriever(DATAROWS_JOSN_PATH).get_df()
    # get the GRoundedSAM annotations

    # for each file, annotation should contain len(classes) rows
    annotations_df = ...
    # merge the two dfs, so that each row in datarows_df has the corresponding annotation for each of the classes
    df = pd.merge(datarows_df, annotations_df, left_index=True, right_index=True)


    uploader = UploadPrelabel()

    for index, row in df.iterrows():
        # get the data row id
        row_id = row['id']
        # get the annotation path
        annotation_path = row['annotation_path']
        # upload the pre label
        for c in classes:
            uploader(row_id=row_id, annotation_path=annotation_path, classname=c, type='segmentation')
