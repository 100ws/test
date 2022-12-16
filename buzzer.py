import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
BUZZER= 23
GPIO.setup(BUZZER, GPIO.OUT)
import time

while True:
    GPIO.output(BUZZER, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(BUZZER, GPIO.LOW)
    time.sleep(2)
    print("done")