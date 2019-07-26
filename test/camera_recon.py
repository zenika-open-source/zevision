
import sys
# Add test option where it gets the path of each laptop and inserts it here
sys.path.insert(0, '/home/ihab/ihabgit/zevision')
import lib.util as predict


predict.launch_camera_feed(record_path='/home/ihab/ihabgit/zevision/test/results/video/webcam_output.avi')
