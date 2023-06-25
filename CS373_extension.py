import cv2
import os
import sys
from pathlib import Path
import re
import CS373_barcode_detection_video

# natural sorting helper
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

# This script take a video, read every frame and try to reveal the position of the barcode with a rectangle, then compiles it to a video
def main():

    # Process commandline inputs
    print("Processing inputs")
    command_line_arguments = sys.argv[1:]
    if (len(command_line_arguments) > 1):
        print("Too many arguments, aborting")
        exit()
    if command_line_arguments != []:
        videoName = command_line_arguments[0]
    else:
        # default file name
        videoName = "example"
    input_video = "videos/"+videoName+".mp4"
    if (not os.path.exists(input_video)):
        print("Cannot find video file")
        exit()

    # Check if folders exist, else create them
    print("Checking folder requirements")
    if not Path("frames").exists():
        Path("frames").mkdir(parents=True, exist_ok=True)
    if not Path("output_frames").exists():
        Path("output_frames").mkdir(parents=True, exist_ok=True)
    if not Path("videos").exists():
        Path("videos").mkdir(parents=True, exist_ok=True)
    if not Path("output_videos").exists():
        Path("output_videos").mkdir(parents=True, exist_ok=True)    

    # Save png frame by frame
    print("Saving video frames")
    vidcap = cv2.VideoCapture(input_video)
    success,image = vidcap.read()
    count = 0
    
    while success:
        cv2.imwrite("frames/frame%d.png" % count, image)      
        success,image = vidcap.read()
        count += 1
    print("Producing frames with barcode detection:")
    dirList = os.listdir("frames")
    dirList = natural_sort(dirList)
    for file in dirList:
        # Call barcode detection on every frame
        print("Processing " + file.rstrip(".png"))
        CS373_barcode_detection_video.main(file.rstrip(".png"))

    
    # Write video to videos folder
    print("Compiling video")
    frame = cv2.imread(os.path.join("output_frames", dirList[0].rstrip(".png") + "_output.png"))
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter("output_videos/"+ videoName+"_output.mp4", fourcc, 30, (width,height)) # 30fps since video recorded on iphone
    for file in dirList:
        video.write(cv2.imread(os.path.join("output_frames", file.rstrip(".png") + "_output.png")))
    video.release()
    
    # Clear input and output frames folder
    print("Clearing frame folders")
    for file in dirList:
        os.remove(os.path.join("frames", file))
        os.remove(os.path.join("output_frames", file.rstrip(".png") + "_output.png"))

    print("Job successfully done")

if __name__ == "__main__":
    main()