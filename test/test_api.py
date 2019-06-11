from flask_testing import TestCase
from unittest import mock, skip
import unittest

import json
import os.path
import pathlib
from io import BytesIO, StringIO
import werkzeug.datastructures

import api


DEFAULT_LISTDIR = ['freddy_mercury', 'config.json', '_hidden']
GET_PREDICTION = [{
    'category': 'winnie_the_pooh',
    'probability': 1.0,
    'box': {
        'top': 0,
        'left': 0,
        'width': 100,
        'height': 100
    }
}]
GET_STATUS = [
    {
        'status': 'failed',
        'path': './logs/model.failed.json',
        'expected': {
            'version': '0.0.0',
            'timestamp': '',
            'recognition_method': 'hog',
            'error': 'something went wrong'
        }
    },
    {
        'status': 'current',
        'path': './logs/model.json',
        'expected': {
            'version': '0.0.0',
            'timestamp': '',
            'recognition_method': 'hog'
        }

    },
    {
        'status': 'pending',
        'path': './logs/model.pending.json',
        'expected': {
            'version': '0.0.0',
            'timestamp': '',
            'recognition_method': 'hog',
            'progress': '100%'
        }
    }
]
GET_CATEGORIES = [
    {
        'category': 'john_cleese',
        'isdir': True,
        'is_malformed': False
    },
    {
        'category': 'long john silver',
        'isdir': True,
        'is_malformed': True
    },
    {
        'category': 'config.json',
        'isdir': False,
        'is_malformed': True
    }
]
ADD_CATEGORY = [
    { # ok
        'data': {
            'categories': (BytesIO(b'''[{
                "category": "ihab_bendidi",
                "data": ["image1.jpg", "image2.jpg"]
            }]'''), 'categories.json'),
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'listdir': {
            'before': ['president_nixon'],
            'after': ['president_nixon', 'ihab_bendidi']
        },
        'category': 'ihab_bendidi',
        'expected': 200
    },
    { # malformed
        'data': {
            'categories': (BytesIO(b'''[{
                "category": "_sylvester_stalone",
                "data": ["image1.jpg", "image2.jpg"]
            }]'''), 'categories.json'),
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'listdir': {
            'before': ['dwayne_the-rock_johnson'],
            'after': ['dwayne_the-rock_johnson']
        },
        'expected': 422
    },
    { # conflict
        'data': {
            'categories': (BytesIO(b'''[{
                "category": "elon_musk",
                "data": ["image1.jpg", "image2.jpg"]
            }]'''), 'categories.json'),
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'listdir': {
            'before': ['elon_musk', 'harry_potter'],
            'after': ['elon_musk', 'harry_potter']
        },
        'expected': 409
    },
    { # missing file or category
        'data': {
            'categories': (BytesIO(b'''[{
                "category": "rolo_tomassi",
                "data": ["image1.jpg", "image3.jpg"]
            }]'''), 'categories.json'),
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'listdir': {
            'before': ['keyser_soze'],
            'after': ['keyser_soze']
        },
        'expected': 404
    },
    { # broken json
        'data': {
            'categories': (BytesIO(b'''[{
                "category": "wade_winston_wilson"
                "data": ["image1.jpg", "image2.jpg"]
            }]'''), 'categories.json'),
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'listdir': {
            'before': ['tyler_durden'],
            'after': ['tyler_durden']
        },
        'expected': 400
    }
]
UPDATE_CATEGORY = [
    {
        'listdir': ['captain_america'],
        'category': 'captain_america',
        'data': {
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'expected': 200
    },
    {
        'listdir': ['agent_scully'],
        'category': 'fox_mulder',
        'data': {
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'expected': 404
    },
    {
        'listdir': ['ethan_hawke'],
        'category': 'stephen_hawcking_',
        'data': {
            'data[]': [
                (BytesIO(b'binary'), 'image1.jpg'),
                (BytesIO(b'binary'), 'image2.jpg')
            ]
        },
        'expected': 422
    },
]
REMOVE_CATEGORY = [
    {
        'listdir': {
            'before': ['mike_portnoy', 'james_labrie', 'john_petrucci', 'john_myung', 'jordan_rudess'],
            'after': ['james_labrie', 'john_petrucci', 'john_myung', 'jordan_rudess']
        },
        'category': 'mike_portnoy',
        'expected': 200
    },
    {
        'listdir': {
            'before': ['lord_marmont'],
            'after': ['lord_marmont']
        },
        'category': 'john_snow',
        'expected': 404
    },
    {
        'listdir': {
            'before': ['captain_america'],
            'after': ['captain_america']
        },
        'category': '_corporal_clegg',
        'expected': 422
    },
]
_IS_MALFORMED_HELPER = [
    ('', True),
    ('_', True),
    ('_one', True),
    ('one_', True),
    ('-', True),
    ('-one', True),
    ('one-', True),
    ('one-two', False),
    ('one_two', False),
    ('one_two-three', False),
    ('one_2_three', True),
    ('one_Two', True),
    ('one__two', True),
    ('one--two', True)
]
_GET_CATEGORIES_HELPER = {
    'listdir': [
        'rick_sanchez',
        'portal_config.json',
        '.mortyrc',
        '.time_trave_stuff',
        '_broken_morty_prototype'],
    'isdir': [True, False, False, True, True],
    'expected': ['rick_sanchez']
}

class Test_API(TestCase):

    def create_app(self):
        return api.app

    def setUp(self):
        app = self.create_app()
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        app.config['DEBUG'] = False
        self.app = app.test_client()
        self.assertEqual(app.debug, False)
        self.predict_patcher = mock.patch('api.predict')
        self.train_model_patcher = mock.patch('api.train_model')
        self.exists_patcher = mock.patch('os.path.exists')
        self.listdir_patcher = mock.patch('os.listdir')
        self.isdir_patcher = mock.patch('os.path.isdir')
        self.move_patcher = mock.patch('shutil.move')
        self.mkdir_patcher = mock.patch('pathlib.Path')
        self.mock_predict = self.predict_patcher.start()
        self.mock_exists = self.exists_patcher.start()
        self.mock_train_model = self.train_model_patcher.start()
        self.mock_move = self.move_patcher.start()
        self.mock_mkdir = self.mkdir_patcher.start()
        self.mock_isdir = self.isdir_patcher.start()
        self.mock_listdir = self.listdir_patcher.start()
        self.mock_listdir.return_value = DEFAULT_LISTDIR
        self.mock_isdir.return_value = [True, False, True]

    def tearDown(self):
        mock.patch.stopall()

    def test_get_prediction(self):
        data = {'data[]': (BytesIO(b'binary'), 'image.jpg')}
        expected = GET_PREDICTION
        self.mock_predict.return_value = GET_PREDICTION
        response = self.client.post('/face_recognition/prediction', data=data)
        self.assertEqual(json.loads(response.data), expected)
        self.mock_predict.assert_called_once()

    def test_get_status(self):
        with mock.patch('json.load') as mock_load:
            for case in GET_STATUS:
                expected = case.get('expected')
                status = case.get('status')
                path = case.get('path')
                with mock.patch('api.open', mock.mock_open(read_data=expected)) as o:
                    mock_load.return_value = expected
                    self.mock_exists.side_effect = lambda p: p == path
                    formatted = {}
                    formatted[status] = expected
                    response = self.app.get('/face_recognition/status')
                    self.assertEqual(json.loads(response.data), formatted)

    def test_get_categories(self):
        expected = [c['category'] for c in GET_CATEGORIES if not c['is_malformed']]
        self.mock_listdir.return_value = [c['category'] for c in GET_CATEGORIES]
        self.mock_isdir.side_effect = [c['isdir'] for c in GET_CATEGORIES]
        response = self.client.get('/face_recognition/categories')
        self.assertEqual(json.loads(response.data), expected)

    def test_add_category(self):
        with mock.patch('werkzeug.datastructures.FileStorage.save'):
            for case in ADD_CATEGORY:
                listdir_before = case.get('listdir', {}).get('before')
                listdir_after = case.get('listdir', {}).get('after')
                data = case.get('data')
                expected = case.get('expected')
                self.mock_listdir.return_value = listdir_before
                self.mock_isdir.return_value = [True for _ in listdir_before]
                if expected == 200:
                    self.mock_mkdir.side_effect = self._add_to_listdir
                response = self.client.put('/face_recognition/categories', data=data)
                self.assertEqual(response.status_code, expected)
                self.mock_train_model.assert_called_once_with(api.RAW_DATA)
                self.assertEqual(api._get_categories(), listdir_after)

    def test_update_category(self):
        with mock.patch('werkzeug.datastructures.FileStorage.save'):
            for i, case in enumerate(UPDATE_CATEGORY):
                self.mock_listdir.return_value = case.get('listdir')
                self.mock_isdir.return_value = [True for _ in case.get('listdir')]
                category = case.get('category')
                response = self.client.patch('/face_recognition/categories/%s' % category,
                    data=case.get('data'))
                expected = case.get('expected')
                actual = response.status_code
                self.assertEqual(response.status_code, expected)
                self.mock_train_model.assert_called_once_with(api.RAW_DATA)
                self.assertEqual(api._get_categories(), case.get('listdir'))
                # Assert files updated

    def test_remove_category(self):
        for i, case in enumerate(REMOVE_CATEGORY):
            category = case.get('category')
            listdir_before = case.get('listdir', {}).get('before')
            expected = case.get('expected')
            self.mock_listdir.return_value = listdir_before
            if expected == 200:
                self.mock_move.side_effect = self._remove_from_listdir
            response = self.client.delete('/face_recognition/categories/%s' % category)
            listdir_after = case.get('listdir', {}).get('after')
            self.assertEqual(response.status_code, expected, '%s' % category)
            self.mock_train_model.assert_called_once_with(api.RAW_DATA)

            self.assertEqual(api._get_categories(), listdir_after)

    def test_is_malformed_helper(self):
        for case in _IS_MALFORMED_HELPER:
            actual = api._is_malformed(case[0])
            expected = case[1]
            self.assertEqual(actual, expected)

    def test_get_category_folder_helper(self):
        result = api._get_category_folder('donald_duck')
        self.assertTrue(result.startswith(api.RAW_DATA))

    def test_get_categories_helper(self):
        self.mock_listdir.return_value = _GET_CATEGORIES_HELPER.get('listdir')
        self.mock_isdir.return_value = _GET_CATEGORIES_HELPER.get('isdir')
        result = api._get_categories()
        expected = _GET_CATEGORIES_HELPER.get('expected')
        self.assertEqual(result, expected)

    def _remove_from_listdir(self, source, destination):
        category = os.path.split(source)[-1]
        if category in self.mock_listdir.return_value:
            self.mock_listdir.return_value.remove(category)
        else:
            category = os.path.split(destination)[-1]
            self.mock_listdir.return_value.remove(category)

    def _add_to_listdir(self, destination):
        category = os.path.split(destination)[-1]
        if not category in self.mock_listdir.return_value:
            self.mock_listdir.return_value.append(category)
        return pathlib.Path
