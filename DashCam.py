# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:06:41 2024

@author: Rob
"""

from pytube import YouTube
import cv2
import tensorflow as tf
import numpy as np

def download_video(url, output_path='downloads/'):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path)
        print(f"Downloaded: {stream.title}")
        return f"{output_path}/{stream.default_filename}"
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None
    


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Display the frame
        cv2.imshow('Video', frame)
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    

def load_model(model_path):
    return tf.saved_model.load(model_path)


def detect_objects(frame, model):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis,...]
    detections = model(input_tensor)
    return detections

def draw_detections(frame, detections):
    for detection in detections['detection_boxes']:
        y1, x1, y2, x2 = detection
        # Convert to pixel coordinates
        height, width, _ = frame.shape
        start_point = (int(x1 * width), int(y1 * height))
        end_point = (int(x2 * width), int(y2 * height))
        color = (0, 255, 0)  # Green
        thickness = 2
        # Draw rectangle
        frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
    return frame

def save_processed_video(input_path, output_path, model):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect_objects(frame, model)
        processed_frame = draw_detections(frame, detections)
        out.write(processed_frame)

    cap.release()
    out.release()
    
    
def main():
    url = input("Enter YouTube URL: ")
    video_path = download_video(url)
    if not video_path:
        return

    model = load_model('path_to_your_model_directory')

    output_path = 'processed_video.avi'
    save_processed_video(video_path, output_path, model)

    print(f"Processed video saved to {output_path}")

if __name__ == "__main__":
    main()