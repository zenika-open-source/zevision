from pathlib import Path
import os
import pickle
import datetime

global encoding_data
global default_encodings
global directory

directory = os.path.dirname(os.path.abspath(__file__))
default_encodings = directory + "/encodings.pickle"




def load_encodings(encodings=default_encodings):
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    try:
        data = pickle.loads(open(encodings, "rb").read())
    except EOFError :
        data = None
    return data


def write_encodings(encodings,file_path=default_encodings):
    print("[INFO] writing encodings...")
    f = open(file_path, "wb")
    data = {"time":datetime.datetime.now(),"data":encodings}
    f.write(pickle.dumps(data))
    f.close()


def update():
    encoding_data = load_encodings(default_encodings)






config = Path(default_encodings)
if config.is_file():
    encoding_data = load_encodings(default_encodings)
else :
    file = open(default_encodings, "x")
    encoding_data = load_encodings(default_encodings)
