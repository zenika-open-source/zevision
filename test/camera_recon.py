
import sys
# Add test option where it gets the path of each laptop and inserts it here
sys.path.insert(0, '/home/ihab/ihabgit/zevision')
import lib.predict as predict


predict.recognize_camera(record_path='/home/ihab/ihabgit/zevision/test/results/video/webcam_output.avi')
