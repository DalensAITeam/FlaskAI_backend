import cv2
import random

from numpy import ndarray


class Animal:


    def run(self, ip_address, main_animal: str) -> tuple[ndarray, int, str]:
        # Corrected IP stream URL
        # url = 'http://192.168.0.94:4747/video'  # Most IP camera apps use /video

        cap = cv2.VideoCapture(ip_address)

        if not cap.isOpened():
            print("Error: Could not open video stream.")
            exit()


        threat_state = "good",

        while True:

            ret, frame = cap.read()
            amount_of_animal_attack = random.randint(0, 50)
            animal_number = random.randint(0, 50)
            amount_of_healthy_animal = random.randint(0, 50)
            amount_of_feeding_animal = random.randint(0, 50)





            yield frame, amount_of_animal_attack, animal_number, amount_of_healthy_animal, amount_of_feeding_animal,threat_state

