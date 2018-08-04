import sys

from yolo import YOLO
from yolo import detect_video

if __name__ == '__main__':
    if len(sys.argv) > 2:
        video_path = sys.argv[1]
        output_path = sys.argv[2]
        detect_video(YOLO(), video_path, output_path)
    elif len(sys.argv) == 2:
        video_path = sys.argv[1]
        detect_video(YOLO(), video_path)
    else:
        detect_video(YOLO()) # webcam capture
