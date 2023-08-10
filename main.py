# This mains load a pre-run labels from GRoundedSAM and uploads them to "covered Baby" project in labelbox, as pre labels
import os
import pandas as pd
from dataset_retriever import DatasetRetriever
from upload_prelabel import UploadPrelabel
from image_retriever import ImageLoader
from annotator import GroundedSam
import tqdm

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    DATAROWS_JOSN_PATH = '/Users/matanb/Downloads/export-result (4).ndjson'
    classes = ["Hand", "Blanket", "Head"]
    # we need to lower the classes because the annotator is lower case, for grinding dino compatibility
    annotator = GroundedSam(classes = [x.lower() for x in classes])

    img_loader = ImageLoader(project='ml-workspace-2', bucket_name='nanobebe_data')
    # get the dataset df
    datarows_df = DatasetRetriever(DATAROWS_JOSN_PATH).get_df()
    uploader = UploadPrelabel('123')

    # data frame for statuof each class for each id
    status_df = pd.DataFrame(columns = classes)
    status_df['id'] = datarows_df['id']
    status_df['path'] = datarows_df.index

    for index, row in tqdm.tqdm(datarows_df.iterrows()):
        # get the data row id
        row_id = row['id']
        url = index

        # row_id = 'cld36xf2v90qx070s9lv8c5aq'
        # url = datarows_df[datarows_df['id'] == row_id].index[0]

        # load the image
        img, image_path = img_loader(url)

        # generate the annotation
        annotation_path = annotator.annotate(img)

        # upload the pre label
        paths = []
        classes_v = []
        for c in classes:
            annotation_path = c.lower() + '.png'
            # annotation files are class + number + .png. look for all the ones for class c
            annot_files = [x for x in os.listdir('.') if x.startswith(c.lower()) and x.endswith('.png')]
            # append to one lsit, not a list of lists
            [paths.append(x) for x in annot_files]
            [classes_v.append(c) for x in annot_files]

        if len(paths) > 0:
            uploader(row_id=row_id, annotation_path=paths, classnames=classes_v, type='segmentation')
            # update at row = id and column = class
            status_df[c][status_df['id'] == row_id] = status_df[c][status_df['id'] == row_id].apply(lambda x: 'uploaded')
            [os.remove(x) for x in paths]
        else:
            status_df[c][status_df['id'] == row_id] = status_df[c][status_df['id'] == row_id].apply(lambda x: 'no annotation')

        status_df.to_csv('status.csv', index=False)
