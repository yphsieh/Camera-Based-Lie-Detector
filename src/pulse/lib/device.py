import cv2, time
#TODO: fix ipcam
#import urllib2, base64
import numpy as np

class ipCamera(object):

    def __init__(self,url, user = None, password = None):
        self.url = url
        auth_encoded = base64.encodestring('%s:%s' % (user, password))[:-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame

class Camera(object):

    def __init__(self, camera = 0, vid=""):
        if vid!="":
            self.cam = cv2.VideoCapture(vid)
            self.video_name = vid
        else:
            self.cam = cv2.VideoCapture(camera)
        #self.cam = cv2.VideoCapture(camera)
        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None

    def get_frame(self):
        if self.valid:
            _,frame = self.cam.read()
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Camera not accessible)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame

    def reset_video(self):
        if hasattr(self, 'video_name'):
            self.cam.set(1, 0)

    def release(self):
        self.cam.release()


class Video(object):
    def __init__(self, vid):
        self.vidname = vid
        self.cam = None
        t0 = 0
        self.start()
        
    def start(self):
        print("Start video")
        if self.vidname == "":
            print("invalid filename!")
            return
            
        self.cam = cv2.VideoCapture(self.vidname)
        fps = self.cam.get(cv2.CAP_PROP_FPS)
        self.frame_count = self.cam.get(cv2.CAP_PROP_FRAME_COUNT)
        #print(fps)
        self.t0 = time.time()
        #print(self.t0)
        self.valid = False
        try:
            resp = self.cam.read()
            self.shape = resp[1].shape
            self.valid = True
        except:
            self.shape = None
    
    
    def stop(self):
        if self.cam is not None:
            self.cam.release()
            print("Stop video")
    
    def get_frame(self):
        if self.valid:
            _,frame = self.cam.read()
            #if frame is None:
                #print("End of video")
                #self.stop()
                #print(time.time()-self.t0)
                #return
            #else:    
                #frame = cv2.resize(frame,(640,480))
        else:
            frame = np.ones((480,640,3), dtype=np.uint8)
            col = (0,256,256)
            cv2.putText(frame, "(Error: Can not load the video)",
                       (65,220), cv2.FONT_HERSHEY_PLAIN, 2, col)
        return frame
        
    def reset_video(self):
        self.cam.set(1, 0)
    

