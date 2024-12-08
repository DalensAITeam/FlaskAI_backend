import time
from pathlib import Path
from typing import Tuple, Any
import cv2
import numpy as np
import supervision as sv
from numpy import ndarray
from supervision import Detections
from ultralytics import YOLO


# Path to the YOLOv8 weights file

class Animal:
    """
    Base class for animal detection using Ultralytics.
    """

    def __init__(self) -> None:
        pass

    def setup_capture(self, ip_address: str) -> cv2.VideoCapture:
        """
        Set up the video capture from the webcam or video file

        Args:
            ip_address: Ip address of the camera

        Returns:
            cv2.VideoCapture: Video capture object.

        """

        capture = cv2.VideoCapture(ip_address)
        frame_width, frame_height = [1280, 720]
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

        return capture

    def setup_model(self, WEIGHTS_PATH: str = "yolov8n.pt") -> YOLO:

        """

        :param WEIGHTS_PATH: The path to trained Yolov8 path
        :return:
        """

        return YOLO(WEIGHTS_PATH)

    def threat_alarm(self, detected_animal_name: str, main_animal: str) -> str:
        """

        :param detected_animal_name: The detected animal (Threat) found on the camera
        :param main_animal: The right animal which we want to detect i.e. The main animal in the farm
        :return: Threat detected or not detected with the name of the threat if  is detected
        """
        print("detected_animal_name",detected_animal_name)
        print("main_animal",main_animal)
        if  main_animal != detected_animal_name:

            return f"Threat '{detected_animal_name}' Detected !!!!"

        else:
            return "No Threat Detected"

    def process_frame(
            self, frame: np.ndarray, model, main_animal: str) -> tuple[ndarray, Detections, str]:

        """

        :param frame: The video frame from the camera
        :param model: The Yolov8 model
        :param main_animal:The right animal which we want to detect i.e. The main animal in the farm

        return:
            frame: The result frame of the detected animal
            detections: The detections from the frame
            alart: Tells if a threat has been detected, tells the threat name that was detected
        """


        # Run YOLOv8 on the frame
        global alart
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)

        bounding_box_annotator = sv.BoundingBoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        labels = [
            model.model.names[class_id]
            for class_id
            in detections.class_id

        ]

        # Annotate the frame with bounding boxes and labels

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        frame = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        for detection in detections:
            animal_name = model.model.names[detection[3]]
            alart = self.threat_alarm(animal_name, main_animal)

        return frame, detections, alart



    def run(self, ip_address, main_animal: str) -> tuple[ndarray, int, str]:
        """
        Run the animal detection application.

        Args:
            main_animal (str): name of the animal to be detected

        Returns:
            None
        """
        print("in")

        capture = self.setup_capture(ip_address)
        model = self.setup_model()
        print("in")

        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame, detections, threat_state = self.process_frame(frame, model, main_animal)
            animal_number = len(detections)

            yield frame, threat_state,animal_number

            # cv2.imshow("Animal Detection", frame)
            # # Exit the loop if the 'Esc' key is pressed
            # if cv2.waitKey(30) == 27:
            #     break



if __name__ == "__main__":

    d = Animal()
    d.run("vid.mp4","bird")

