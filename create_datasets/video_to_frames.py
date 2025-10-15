import os
import cv2
input_filename = '../datasets/video/video1.mp4'
output_path='../datasets/video/video1/'
vid = cv2.VideoCapture(input_filename)

os.makedirs(output_path, exist_ok=True)

count, success = 0, True
while success:
    success, image = vid.read() # Read frame
    if success:
        out_filename = output_path + f"frame{count:08d}.jpg"
        cv2.imwrite(out_filename, image) # Save frame
        count += 1

vid.release()