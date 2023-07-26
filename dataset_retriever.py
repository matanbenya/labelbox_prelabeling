import json
import pandas as pd

class DatasetRetriever:
    def __init__(self, json_path):
        self.json_path = json_path
        self.raw_data = self.read_json()

    def read_json(self):
        with open(self.json_path) as f:
            data = f.readlines()
            data = [json.loads(x) for x in data]
        return data

    def get_ids(self):
        ids = [x['data_row']['id'] for x in self.raw_data]
        return ids

    def get_paths(self):
        paths = [x['data_row']['row_data'] for x in self.raw_data]
        return paths

    def get_df(self):
        paths = self.get_paths()
        ids = self.get_ids()
        df = pd.DataFrame({'id': ids, 'path': paths})
        # set path as index
        df.set_index('path', inplace=True)
        return df

    def get_annotation_paths(self):
        raise NotImplementedError
