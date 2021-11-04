'''
usage: python taping.py --subject [name] --data [data_path]
'''
import os
import cv2
import pyaudio
import wave
import argparse
import time

class VideoRecorder(object):
	def __init__(self, args):

		self.subject = args.subject
		self.first_vid = False
		self.q = 0
		self.filename = self.subject + '_q' + '{:02d}'.format(self.q)
		self.tStart = time.time()

		self.cap = cv2.VideoCapture(0)
		#self.cap.set(3,640)
		#self.cap.set(4,480)

		self.fps = 30
		self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

		self.sz = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
		self.out = cv2.VideoWriter(os.path.join(args.data, args.subject, self.filename + ".mp4"), self.fourcc, self.fps, self.sz)

		self.rate = 44100
		self.frames_per_buffer = 1500
		self.channels = 1
		self.format = pyaudio.paInt16
		self.audio = pyaudio.PyAudio()
		self.stream = self.audio.open(format=self.format, channels=self.channels, rate=self.rate, input=True, frames_per_buffer = self.frames_per_buffer)
		self.audio_frames = []

	def run(self):
		self.tStart = time.time()
		if self.first_vid: print("\nRecording..." + self.filename)			
		self.out = cv2.VideoWriter(os.path.join(args.data, args.subject, self.filename + ".mp4"), self.fourcc, self.fps, self.sz)
		self.audio_frames = []


		while(self.cap.isOpened()):
			_, frame = self.cap.read()
			frame = cv2.flip(frame, 1)
			self.out.write(frame)

			if not self.first_vid:
				cv2.putText(frame, "Press 's' to start recording", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 100))
				cv2.putText(frame, "Press 'q' to quit recording", (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 100))
			else :
				cv2.putText(frame, "Recording..." + self.filename , (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 100))
				cv2.putText(frame, "Press 's' to record next video", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 100))
				cv2.putText(frame, "Press 'q' to quit recording", (10, 75), cv2.FONT_HERSHEY_PLAIN, 1.5, (100, 255, 100))
			
			cv2.imshow('frame',frame)

			data = self.stream.read(self.frames_per_buffer, exception_on_overflow = False) 
			self.audio_frames.append(data)

			key = cv2.waitKey(10)
			if key & 0xFF == ord('q') or key & 0xFF == ord('s') :
				if self.first_vid : print("time of question " + str(self.q) + ": " +str(time.time() - self.tStart))
				self.out.release()
				self.stream.stop_stream()

				waveFile = wave.open(os.path.join(args.data, args.subject, self.filename + ".wav"), 'wb')
				waveFile.setnchannels(self.channels)
				waveFile.setsampwidth(self.audio.get_sample_size(self.format))
				waveFile.setframerate(self.rate)
				waveFile.writeframes(b''.join(self.audio_frames))
				waveFile.close()

				if not self.first_vid: self.first_vid = True
				else :
					self.q += 1
					self.filename = self.subject + '_q' + '{:02d}'.format(self.q)

				if key & 0xFF == ord('q'): return False
				elif key & 0xFF == ord('s'): return True


if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='Data Recording.')
	parser.add_argument('--data', default='./data_video', help='data directory to save the videos')
	parser.add_argument('--subject', default='who', help='Enter subject name')
	args = parser.parse_args()

	if not os.path.exists(os.path.join(args.data, args.subject)):
		os.makedirs(os.path.join(args.data, args.subject))

	record = True

	vid = VideoRecorder(args)
	
	while (record):
		vid.stream.start_stream()
		record = vid.run()
	
	vid.cap.release()
	cv2.destroyAllWindows()
	vid.stream.close()
	vid.audio.terminate()

	for i in range(vid.q):
		filename = args.subject + '_q' + '{:02d}'.format(i)
		bashCommand = "ffmpeg -i "+os.path.join(args.data, args.subject, filename + ".mp4")+" -i "+ os.path.join(args.data, args.subject, filename + ".wav") + " -c:v copy -c:a aac " + os.path.join(args.data, args.subject, filename + "_fin.mp4")
		os.system(bashCommand)
		os.remove(os.path.join(args.data, args.subject, filename + ".mp4"))
		os.remove(os.path.join(args.data, args.subject, filename + ".wav"))

