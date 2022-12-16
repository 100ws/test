#!/usr/bin/env python3

import os 
import time


while True:
    time.sleep(1)
    with open("/home/pi/Desktop/final_drowsiness/reading_data.text","r") as reading_data:
        data=reading_data.read()
        data=int(data)
        print(data)
        print(type(data))

    # with open("file_running.text","r") as read_file_running:
    #     read_file_running.read()
    # read_file_running=read_file_running


    if data==0:
        os.system("sudo python3 /home/pi/Desktop/final_drowsiness/system_drowsy.py")
        # while True:
        #     time.sleep(1)
        #     print(1)

    if data==1:
        # while True:
        #     time.sleep(1)
        #     print(2)
        os.system("sudo python3 /home/pi/Desktop/final_drowsiness/drowsy.py")

    if data==2:
        os.system("sudo python3 /home/pi/Desktop/final_drowsiness/training/train_model.py")


    # elif data==1:
    #     os.system("sudo /home/pi/Desktop/final_drowsiness/training/train_model.py")
    # elif data==2:
    #     os.system("sudo /home/pi/Desktop/final_drowsiness/system_drowsy.py")
