import cv2
import supervision as sv
from ultralytics import YOLO
from roboflow import Roboflow
import APIk

rf = Roboflow(api_key= APIk.ApiK.get_APIkey())
project = rf.workspace("bubbles-dm1p6").project("bubble-jgyy7")
version = project.version(1)
dataset = version.download("yolov11")

