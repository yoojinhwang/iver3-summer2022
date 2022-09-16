import yaml
import pandas as pd
import os

datasets_path = '../data/datasets.yaml'
dataset_name = 'tag78_swimming_test_1'
datasets = yaml.safe_load(open(datasets_path, 'r'))
dataset_obj = datasets[dataset_name]

def get_path(relpath):
    return os.path.join(os.path.split(datasets_path)[0], relpath)

tag_id = dataset_obj['tag']
hydrophone_pds = []
for _, hydrophone_dict in dataset_obj['hydrophones'].items():
    print(hydrophone_dict['path'])
    if type(hydrophone_dict['path']) is list:
        hydrophone_pds = hydrophone_pds + [pd.read_csv(get_path(path)) for path in hydrophone_dict['path']]
    else:
        hydrophone_pds.append(pd.read_csv(get_path(hydrophone_dict['path'])))
hydrophone_data = pd.concat(hydrophone_pds)
hydrophone_data = hydrophone_data[hydrophone_data['tag_id'] == tag_id]
hydrophone_data['datetime'] = pd.to_datetime(hydrophone_data['datetime'])
hydrophone_data.sort_values('datetime', inplace=True)

tag_coords_obj = dataset_obj['tag_coords']
if 'path' in tag_coords_obj:
    tag_coords_pd = pd.read_csv(tag_coords_obj['path'])
else:
    tag_latitude, tag_longitude = tag_coords_obj['latitude'], tag_coords_obj['longitude']
    hydrophone_data['tag_latitude'] = [tag_latitude] * len(hydrophone_data)
    hydrophone_data['tag_longitude'] = [tag_longitude] * len(hydrophone_data)
