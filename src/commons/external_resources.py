import json

import gridfs
from pymongo import MongoClient
from sacred.observers import MongoObserver
from visdom import Visdom


def load_conf(path):
    with open(path) as file:
        conf = json.load(file)
    return conf


###
# Mongodb
###


def get_mongo_connection_url(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)
    db_user = '{}:{}'.format(mongo_conf['user'], mongo_conf['passwd'])
    db_host = '{}:{}'.format(mongo_conf['host'], mongo_conf['port'])
    auth_db = mongo_conf.get('auth_db', mongo_conf['db'])
    return 'mongodb://{}@{}/{}'.format(db_user, db_host, auth_db)


def get_mongo_db(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    connection_url = get_mongo_connection_url(mongo_conf)
    return MongoClient(connection_url)[mongo_conf['db']]


def get_mongo_collection(mongo_conf=None, mongo_path=None, collection=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)
    db = get_mongo_db(mongo_conf)

    if collection is None:
        collection = mongo_conf['collection']

    return db[collection]


def get_mongo_obs(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    db_url = get_mongo_connection_url(mongo_conf)
    return MongoObserver.create(url=db_url, db_name=mongo_conf['db'], collection=mongo_conf['collection'])


def get_gridfs(mongo_conf=None, mongo_path=None):
    if mongo_conf is None:
        mongo_conf = load_conf(mongo_path)

    return gridfs.GridFS(get_mongo_db(mongo_conf))


###
# Visdom
###

def get_visdom_conf(visdom_path=None, **conf_updates):
    visdom_conf = dict(raise_exceptions=True)

    if visdom_path is not None:
        visdom_conf.update(load_conf(visdom_path))
    else:
        visdom_conf.update(server='http://localhost', port=8097)

    visdom_conf.update(conf_updates)
    return visdom_conf

def get_visdom(visdom_path=None, **conf_updates):
    return Visdom(**get_visdom_conf(visdom_path, **conf_updates))
