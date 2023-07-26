import labelbox as lb
import labelbox.data.annotation_types as lb_types
import uuid


class UploadPrelabel:
    def __init__(self):
        API_KEY = r'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGpvYnBlY2kwNHk0MDd5cTViNGIzMzAwIiwib3JnYW5pemF0aW9uSWQiOiJjbGQxdjY4cmMwZHV0MDcyNzg5bHgydGkxIiwiYXBpS2V5SWQiOiJjbGtpYWlrZXoxY2hsMDczZmdvemwyZ2JoIiwic2VjcmV0IjoiYzFjMWFhNDY0YjAxYWY4M2I2MDI1Y2JmZGUzZTM1MWQiLCJpYXQiOjE2OTAyODkxNDYsImV4cCI6MjMyMTQ0MTE0Nn0.fPe3UwQl3lvruxoE7BBw461o050S8zlJ0V1kvr3824I'
        self.client = lb.Client(API_KEY)
        project = self.client.get_project('cljvb348h00ef07xf2t4u3fah')
        self.project = project
        pass

    def assign_global_key(self, row_id = None):
        self.data_row = self.client.get_data_row(row_id)
        # assign global key
        global_key = str(uuid.uuid4())
        global_key_data_row_inputs = [
            {"data_row_id": self.data_row.uid, "global_key": global_key}
        ]
        self.client.assign_global_keys_to_data_rows(global_key_data_row_inputs)
        return global_key

    def get_row_metadata(self):
        # get data row metadata
        height = self.data_row.media_attributes['height']
        width = self.data_row.media_attributes['width']
        return height, width

    # type can be only segementation or bbox
    def get_annotation(self, annotation_path = None, classname = None, type = 'segmentation'):
        # Identifying what values in the numpy array correspond to the mask annotation
        annotation = []
        if type == 'segmentation':
            color = (255, 255, 255)
            mask_data = lb_types.MaskData(file_path=annotation_path)

            annotation = lb_types.ObjectAnnotation(
                name=classname,  # must match your ontology feature"s name
                value=lb_types.Mask(mask=mask_data, color=color),
            )

        else:
            # Python annotation
            bbox_annotation = lb_types.ObjectAnnotation(
                name="Covered Baby",  # must match your ontology feature"s name
                value=lb_types.Rectangle(
                    start=lb_types.Point(x=350, y=200),  # x = left, y = top
                    end=lb_types.Point(x=500, y=450),  # x= left + width , y = top + height
                ))
        return annotation


    def __call__(self, row_id = None, annotation_path = None, classname = 'Blanket', type = 'segmentation'):


        global_key = self.assign_global_key(row_id)

        # get the annotation
        mask_annotation = self.get_annotation(annotation_path = annotation_path, classname = classname, type = type)
        labels = []
        annotations = [
            mask_annotation,
        ]
        labels.append(
            lb_types.Label(data=lb_types.ImageData(global_key=global_key),
                           annotations=annotations))

        # Upload MAL label for this data row in project
        upload_job = lb.MALPredictionImport.create_from_objects(
            client=self.client,
            project_id=self.project.uid,
            name="mal_job" + str(uuid.uuid4()),
            predictions=labels)

        print(f"Errors: {upload_job.errors}", )
        print(f"Status of uploads: {upload_job.statuses}")

        pass




