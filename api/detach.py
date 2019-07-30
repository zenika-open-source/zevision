"""Run a long running procedure is a separate thread."""

# TODO: Multiple instances of the same process should not be supported.


from datetime import datetime
import subprocess
import os.path
import json
import os

from lib.util import train_face_model


def wrap_train_model(RAW_DATA, LOG_PATH):
    failed = os.path.join(LOG_PATH, 'model.failed.json')
    pending = os.path.join(LOG_PATH, 'model.pending.json')
    ready = os.path.join(LOG_PATH, 'model.json')

    current_model = _get_model_info(LOG_PATH)
    with open(pending, 'w') as fh:
        # current_model['progress'] = 0
        print(json.dumps(current_model, ensure_ascii=False, indent=2), file=fh)
    if os.path.exists(failed):
        os.remove(failed)

    try:
        res = subprocess.run(
            ['python3',
            '-m',
            'lib.util',
            '--data',
            RAW_DATA]
        )
        """
        with open(pending, 'w') as fh:
            for progress in cmd(param):
                model_info['progress'] = progress
                print(json.dumps(model_info, ensure_ascii=False, indent=2), file=fh)
                fh.seek(0)
        """
        # TODO: reload model
        with open(ready, 'w') as fh:
            # del current_model['progress']
            current_model['timestamp'] = datetime.now().isoformat()
            print(json.dumps(current_model, ensure_ascii=False, indent=2), file=fh)
    except Exception as e:
        with open(failed, 'w') as fh:
            current_model['timestamp'] = datetime.now().isoformat()
            current_model['error'] = str(e)
            # del current_model['progress']
            print(json.dumps(current_model, ensure_ascii=False, indent=2), file=fh)
    finally:
        if os.path.exists(pending):
            os.remove(pending)
    return True


def _bump(version='0.0.0'):
    """
    >>> _bump()
    '0.0.1'
    >>> _bump('0.0.3')
    '0.0.4'
    >>> _bump('1.0.5')
    '1.0.6'
    """
    major, minor, patch = version.split('.')
    patch = str(int(patch) + 1)
    return '.'.join([major, minor, patch])

def _get_version(log_path):
    for path in [
        os.path.join(log_path, 'model.failed.json'),
        os.path.join(log_path, 'model.json')
    ]:
        if os.path.exists(path):
            with open(path) as fh:
                return json.load(fh).get('version')
    return '0.0.1'

def _get_model_info(log_path):
    return {
        'version': _bump(_get_version(log_path)),
        'timestamp': datetime.now().isoformat()
    }


if __name__ == '__main__':

    import doctest
    doctest.testmod()
