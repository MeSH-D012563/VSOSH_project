import requests
import time
from picamera import PiCamera
from picamera.array import PiRGBArray
import cv2
import RPi.GPIO as GPIO

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(21, GPIO.OUT)
GPIO.setup(22, GPIO.OUT)
GPIO.setup(23, GPIO.OUT)
GPIO.setup(24, GPIO.OUT)

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(320, 240))

time.sleep(0.1)


def control_motors(command, duration):
    if command == 'forward':
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.LOW)
    elif command == 'backward':
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.HIGH)
    elif command == 'left':
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.HIGH)
        GPIO.output(23, GPIO.HIGH)
        GPIO.output(24, GPIO.LOW)
    elif command == 'right':
        GPIO.output(21, GPIO.HIGH)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.HIGH)
    elif command == 'stop':
        GPIO.output(21, GPIO.LOW)
        GPIO.output(22, GPIO.LOW)
        GPIO.output(23, GPIO.LOW)
        GPIO.output(24, GPIO.LOW)

    time.sleep(duration)
    control_motors('stop', 0)


def execute_route(route_name):
    response = requests.post('http://SERVER_IP:5000/execute_route', json={'route_name': route_name})
    commands = response.json().get('commands', [])
    for command in commands:
        parts = command.split()
        if len(parts) == 2:
            cmd, duration = parts
            control_motors(cmd, float(duration))


def send_image_to_server(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': (image_path, f, 'image/jpeg')}
        response = requests.post('http://SERVER_IP:5000/upload', files=files)
        return response.json()


def process_video():
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        rawCapture.truncate(0)
        cv2.imwrite('current_frame.jpg', image)
        response = send_image_to_server('current_frame.jpg')
        if response.get('status') == 'success':
            print("Изображение успешно отправлено на сервер")
        time.sleep(0.1)


if __name__ == '__main__':
    try:
        threading.Thread(target=process_video).start()
        while True:
            response = requests.get('http://SERVER_IP:5000/signal_robot')
            if response.status_code == 200:
                route_name = response.json().get('route_name')
                execute_route(route_name)
            time.sleep(1)
    except KeyboardInterrupt:
        camera.close()
        GPIO.cleanup()
