

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import pandas as pd
import numpy as np
import imutils
import time
import glob
import dlib
import cv2
import os

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
ap.add_argument("-v_d", "--video_dir", type=str, default="../",
    help="path to input video files")
ap.add_argument("-t", "--threshold", type = float, default=0.27,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--frames", type = int, default=2,
	help="the number of consecutive frames the eye must be below the threshold")
ap.add_argument("-sub", "--subject", type=str)

def main() :
    args = vars(ap.parse_args())
    EYE_AR_THRESH = args['threshold']
    EYE_AR_CONSEC_FRAMES = args['frames']
    selected_cam = 0
    cameras = []
    ears = []

    # initialize the frame counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    # start the video stream thread
    print("[INFO] starting video stream thread...")
    print("[INFO] print q to quit...")
    if args['video_dir'] is None:
        if args['video'] == "camera":
            vs = VideoStream(src=0).start()
            fileStream = False
    else:
        # vs = FileVideoStream(args["video"]).start()
        fileStream = True
        video_names = glob.glob(args['video_dir'] + '/*.mp4')
        print(video_names)
        video_names.sort()
        for i in range(len(video_names)):
            camera = FileVideoStream(video_names[i])
            cameras.append(camera)
        vs = cameras[selected_cam].start()

    time.sleep(1.0)
    
    subject = args['subject']
    if not os.path.isdir(os.path.join('./data_blink', subject)): os.mkdir(os.path.join('./data_blink', subject))
    # loop over frames from the video stream
    while True:
        # print(video_names[selected_cam], vs.more())
        if fileStream and not vs.more():
            if args['video_dir'] is None:
                break

            else:
                vs.stop()

                pos = video_names[selected_cam].find("_q")
                question_number = video_names[selected_cam][pos + 2:pos + 4]
                fn = os.path.join('./data_blink', subject, subject + '_' + question_number)
                data = np.array(ears)
                df = pd.DataFrame(data=data, columns=['EAR'])
                df.to_csv(fn + ".csv")

                if selected_cam == 0 :
                    EYE_AR_THRESH = np.average(ears)
                    print('threshold: ', EYE_AR_THRESH)
                ears = []

                # Change to the next video (in the next camera)
                selected_cam += 1

                if selected_cam >= len(video_names): break
                
                vs = cameras[selected_cam].start()
                print('Changing to next video...{}'.format(video_names[selected_cam]))
    
        # print(vs)
        try:
            frame = vs.read()
            frame = imutils.resize(frame, width=450)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                ears.append(ear)

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                	COUNTER += 1

                # otherwise, the eye aspect ratio is not below the blink
                # threshold
                else:
                	# if the eyes were closed for a sufficient number of
                	# then increment the total number of blinks
                	if COUNTER >= EYE_AR_CONSEC_FRAMES:
                		TOTAL += 1

                	# reset the eye frame counter
                	COUNTER = 0

                # draw the total number of blinks on the frame along with
                # the computed eye aspect ratio for the frame
                cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        	# show the frame
            cv2.imshow("Frame", frame)

        except: pass
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
        	break
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
if __name__ == '__main__' :
    main()
