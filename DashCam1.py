# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:10:07 2024

@author: Rob
"""

import cv2
import torch
from pytube import YouTube
import os

# Step 1: Function to download YouTube video
def download_video(youtube_url, save_path="downloads"):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension="mp4").first()
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f"{yt.title}.mp4")
    stream.download(output_path=save_path, filename=f"{yt.title}.mp4")
    print(f"Video downloaded to {file_path}")
    return file_path

# Step 2: Load pre-trained model (YOLOv5 example)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Download YOLOv5 small model
    return model

# Step 3: Process video frames
def process_video(input_video_path, output_video_path, model):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print("Processing video...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for the model
        results = model(frame[..., ::-1])  # YOLOv5 ex
        pects RGB
        processed_frame = results.render()[0]  # Overlay results on frame

        # Write the frame to the output video
        out.write(processed_frame)

    cap.release()
    out.release()
    print(f"Processed video saved at {output_video_path}")

# Step 4: Main program
if __name__ == "__main__":
    youtube_url = input("Enter YouTube URL: ")
    downloaded_video = download_video(youtube_url)

    # Load pre-trained model
    model = load_model()

    # Process the video
    output_path = "processed_video.mp4"
    process_video(downloaded_video, output_path, model)

    print("Program complete!")