from imutils.video import VideoStream
#from imutils.video import FPS
import predict as predict


# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
time.sleep(2.0)

i = 0

# start the FPS throughput estimator
#fps = FPS().start()
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    i++
    frame = vs.read()
    image_path = "resources/video/"+ i + ".jpg"
    cv2.imwrite(image_path, frame)
    response = predict.recognize_face(image_path)
    for (i,r) in enumerate(response):
        print("\n \n the number ",i+1," prediction for ",image," is  :   ",r)
        box = dlib.rectangle(r["box"][3], r["box"][0], r["box"][1], r["box"][2])
        top = box.top()
        right = box.right()
        bottom = box.bottom()
        left = box.left()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, r["category"], (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
