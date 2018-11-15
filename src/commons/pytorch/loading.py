import os
import shutil
import tempfile
import torch

from src.commons import external_resources
import logging

logger = logging.getLogger(__name__)


def get_state(name, mongo_path):
    mongo_collection = external_resources.get_mongo_collection(mongo_path=mongo_path)
    mongo_fs = external_resources.get_gridfs(mongo_path=mongo_path)

    file_id = get_file_id(name, mongo_collection)
    return get_artifact(file_id, mongo_fs)


def get_file_id(artifact_name, collection):
    """
    :param artifact_name:
    :param collection:
    :return: the corresponding file_id if artifact_name exists, None otherwise
    """
    exp = collection.find_one({'artifacts.name': artifact_name}, {'artifacts': 1})
    if exp is None:
        return None

    for elt in exp['artifacts']:
        if elt['name'] == artifact_name:
            return elt['file_id']


def get_artifact(object_id, fs, device='cpu'):
    bin = fs.get(object_id)  # Gridout object

    # Strange way to have the GridOut object as a python File
    temp_path = tempfile.mkdtemp()
    temp_file = os.path.join(temp_path, 'salut')
    with open(temp_file, 'wb') as f:
        f.write(bin.read())
        obj = torch.load(temp_file, map_location=device)
    shutil.rmtree(temp_path)
    return obj
