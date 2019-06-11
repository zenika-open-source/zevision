"""API server for Face Recognition service.

Maintainer: egor-jerome.akhanov@zenika.com

Github: https://github.com/Zenika/zihub
"""

from werkzeug.utils import secure_filename
from logging.config import dictConfig
from gevent.pywsgi import WSGIServer
from flask import Flask, request
import threading
import tempfile
import logging
import pathlib
import shutil
import json
import os

from api.detach import wrap_train_model
from lib.predict import recognize_face as predict


# TODO: 423: Locked - another training is runing.


RAW_DATA = 'data/raw'
LOG_PATH = 'logs'
LOG_NAME = 'face_recognition_api.log'

LOGGER_CONFIG = {
    'version': 1,
    'formatters': {
        'default': {
            'format': '%(asctime)s\t%(levelname)s\t%(message)s',
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(LOG_PATH, LOG_NAME),
            'maxBytes': 10 ** 8,
            'mode': 'a',
            'formatter': 'default'
        }
    },
    'root': {
        'level': 'DEBUG',
        'handlers': ['file']
    }
}
dictConfig(LOGGER_CONFIG)

app = Flask(__name__)


# Helpers

def abort(reason='', status=500):
    app.logger.error(reason)
    response = {
        'error': reason,
        'status': status
    }
    return _json_parse(response)

def _get_categories(root=RAW_DATA):
    if not os.path.exists(root):
        return []
    return [category for category in os.listdir(root) \
        if os.path.isdir(os.path.join(root, category))]

def _indent(param='pretty'):
    if __name__ == 'api.server':
        return 2
    if param in request.args and request.args.get(param) != 'false':
        return 2

def _get_category_folder(category, root=RAW_DATA):
    return os.path.join(root, category)

def _json_parse(data):
    return json.dumps(data, ensure_ascii=False, indent=_indent())


# Training wrapper

def train_model():
    threading.Thread(target=wrap_train_model, args=(RAW_DATA, LOG_PATH)).start()


# Routes

@app.route('/face_recognition/prediction', methods=['POST'])
def get_prediction():
    """Get model predictions.
    Only the best prediction for each detected face should be returned.
    """

    app.logger.debug('Getting a prediction.')
    upload = request.files['data']
    destination = os.path.join(tempfile.gettempdir(), upload.filename)
    try:
        upload.save(destination)
        prediction = predict(destination)
        os.remove(destination)
        return _json_parse(prediction)
    except FileNotFoundError as e:
        # This is probably not a good place to trigger FileNotFound
        # since the model should only be loaded once
        return abort('Unable to locate model data: %s.' % str(e), 503)
    except Exception as e:
        return abort('Unable to retrieve predictions: %s.' % str(e), 500)


@app.route('/face_recognition/status', methods=['GET'])
def get_status():
    """Return model status, could be used to track training failures
    and model update progress.
    """

    app.logger.debug('Getting model status.')
    logs = {
        'failed': os.path.join(LOG_PATH, 'model.failed.json'),
        'pending': os.path.join(LOG_PATH, 'model.pending.json'),
        'current': os.path.join(LOG_PATH, 'model.json')
    }
    status = {}
    for level, path in logs.items():
        if os.path.exists(path):
            with open(path) as fh:
                status[level] = json.load(fh)
    return _json_parse(status)


@app.route('/face_recognition/categories', methods=['GET'])
def get_categories():
    """List categories available to the model. In case a training is still running,
    new categories should be listed as well.
    """

    app.logger.debug('Getting category list.')
    categories = _get_categories()
    return _json_parse(categories)


@app.route('/face_recognition/retrain', methods=['GET'])
def retrain():
    """Retrain model without any data change."""

    app.logger.debug('Retrain the model.')
    try:
        train_model()
    except Exception as e:
        return abort('Unable to run model training: %s.' % str(e), 500)
    return '', 204


@app.route('/face_recognition/categories', methods=['PUT'])
def add_category():
    """Add new categories data to the model."""

    file_2_category = {}
    existing_categories = _get_categories()
    add_categories = []

    app.logger.debug('Add category.')
    app.logger.debug('Validate categories.json and build file-to-category maps.')
    for upload in request.files.getlist('categories'):
        try:
            meta = json.loads(upload.read().decode('utf-8'))
            for data_class in meta:
                category = data_class['category']
                if category in existing_categories:
                    return abort('Category already exists: %s' % category, 409)
                for file_name in data_class['data']:
                    file_2_category[file_name] = category
                add_categories.append(category)
        except (KeyError, json.JSONDecodeError) as e:
            return abort('Unable to parce JSON data: %s.' % str(e), 400)

    app.logger.debug('Make sure mappings are provided for all files')
    for upload in request.files.getlist('data'):
        file_name = upload.filename
        if not file_name in file_2_category:
            return abort('Unable to map file to category: %s.' % file_name, 404)

    app.logger.debug('Process images.')
    for upload in request.files.getlist('data'):
        file_name = upload.filename
        try:
            category_folder = _get_category_folder(file_2_category[file_name])
            sample = os.path.join(category_folder, secure_filename(file_name))
            pathlib.Path(category_folder).mkdir(parents=True, exist_ok=True)
            upload.save(sample)
        except Exception as e:
            for category in add_categories:
                category_folder = _get_category_folder(category_name)
                shutil.rmtree(category_folder, ignore_errors=True)
            return abort('Unable to process uploaded file: %s: %s' % (file_name, str(e)), 500)

    try:
        app.logger.debug('Update the model.')
        train_model()
    except Exception as e:
        for category in add_categories:
            category_folder = _get_category_folder(category)
            shutil.rmtree(category_folder, ignore_errors=True)
        return abort('Unable to run model training: %s.' % str(e), 500)
    return '', 201


@app.route('/face_recognition/categories/<category>', methods=['PATCH'])
def update_category(category):
    """Add new data to an existing category."""

    app.logger.debug('Update category.')
    app.logger.debug('Validate categories.json.')
    if not category in _get_categories():
        return abort('Unable to locate specified category: %s.' % category, 404)

    app.logger.debug('Create backup files.')
    category_folder = _get_category_folder(category)
    tmpdir = tempfile.mkdtemp()
    shutil.copytree(category_folder, os.path.join(tmpdir, category))

    app.logger.debug('Process images.')
    for upload in request.files.getlist('data'):
        file_name = upload.filename
        try:
            sample = os.path.join(category_folder, secure_filename(file_name))
            upload.save(sample)
        except Exception as e:
            shutil.rmtree(category_folder, ignore_errors=True)
            shutil.move(os.path.join(tmpdir, category), category_folder)
            shutil.rmtree(tmpdir, ignore_errors=True)
            return abort('Unable to process uploaded file: %s: %s' % (file_name, str(e)), 500)

    try:
        app.logger.debug('Update the model.')
        train_model()
    except Exception as e:
        shutil.rmtree(category_folder, ignore_errors=True)
        shutil.move(os.path.join(tmpdir, category), category_folder)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return abort('Unable to run model training: %s.' % str(e), 500)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return '', 204


@app.route('/face_recognition/categories/<category>', methods=['DELETE'])
def remove_category(category):  # Remove processed images
    """Remove one category at a time from the model. All data related to this category,
    including processed images, should be erased.
    """

    app.logger.debug('Remove category.')
    app.logger.debug('Validate categories.json.')
    if not category in _get_categories():
        return abort('Unable to locate specified category: %s' % category, 404)

    app.logger.debug('Process images.')
    tmpdir = tempfile.mkdtemp()
    try:
        category_folder = _get_category_folder(category)
        shutil.move(category_folder, tmpdir)
        app.logger.debug('Category %s deleted. Update the model.' % category)
        train_model()
    except Exception as e:
        shutil.move(os.path.join(tmpdir, category), category_folder)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return abort('Unable to delete category: %s: %s.' % (category, str(e)), 500)

    shutil.rmtree(tmpdir, ignore_errors=True)
    return '', 204


if __name__ == '__main__':

    from sys import stderr
    import argparse

    DEFAUL_PORT = 5000

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-p', '--port', default=DEFAUL_PORT, type=int,
        help='API server port.')
    parser.add_argument('-d', '--data', default=RAW_DATA,
        help='Path to data folder. Default: %s.' % RAW_DATA)
    parser.add_argument('-l', '--logs', default=LOG_PATH,
        help='Path to logs folder. Default: %s.' % LOG_PATH)
    parser.add_argument('--log_name', default=LOG_NAME,
        help='Server log file name. Default: %s.' % LOG_NAME)

    args = parser.parse_args()

    LOG_PATH = args.logs
    RAW_DATA = args.data
    LOG_NAME = args.log_name

    LOGGER_CONFIG['handlers']['file']['filename'] = os.path.join(LOG_PATH, LOG_NAME)
    dictConfig(LOGGER_CONFIG)

    print('Listening on port %d' % args.port, file=stderr)
    http_server = WSGIServer(('', args.port), app)
    http_server.serve_forever()
