import os
import requests
import threading
import telebot
import numpy as np
import time
import cv2

bot = telebot.TeleBot('YOUR_BOT_TOKEN')

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

start_camera = False
user_chat_id = None

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

routes_folder = 'routes'
if not os.path.exists(routes_folder):
    os.makedirs(routes_folder)


def process_image(image_path):
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == "person":
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    person_detected = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            person_detected = True

    if person_detected:
        print("Человек обнаружен!")
        cv2.imwrite('detected_person.jpg', image)
        if user_chat_id:
            bot.send_photo(user_chat_id, photo=open('detected_person.jpg', 'rb'))
            bot.send_message(user_chat_id, 'Был обнаружен человек. Для продолжения введите /restart')
        return True
    return False


def process_video():
    global start_camera, user_chat_id
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Не удалось открыть веб-камеру")
        return

    time.sleep(0.1)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not start_camera:
            break

        frame_count += 1
        if frame_count % 3 == 0:
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5 and classes[class_id] == "person":
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            person_detected = False

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    person_detected = True

            if person_detected:
                print("Человек обнаружен!")
                cv2.imwrite('detected_person.jpg', frame)
                start_camera = False
                if user_chat_id:
                    bot.send_photo(user_chat_id, photo=open('detected_person.jpg', 'rb'))
                    bot.send_message(user_chat_id, 'Был обнаружен человек. Для продолжения введите /restart')
                    send_signal_to_robot()
                break

    cap.release()
    cv2.destroyAllWindows()


def send_signal_to_robot():
    response = requests.post('http://ROBOT_IP:5000/start_route', json={'route_name': 'example_route'})
    if response.status_code == 200:
        print("Сигнал отправлен роботу для проверки")
    else:
        print("Ошибка при отправке сигнала роботу")


def start(message):
    global user_chat_id
    user_chat_id = message.chat.id
    bot.send_message(user_chat_id, 'Привет пользователь!\nДоступные команды можно узнать через /help')


def help(message):
    bot.send_message(message.chat.id,
                     'Доступные команды:\n/start - Начать работу с ботом\n/start_work - Включить камеру и начать поиск человека\n/help - Показать список доступных команд\n/restart - Перезапустить камеру\n/route_camera1 - Создать маршрут охвата одной камеры')


def start_work(message):
    global start_camera, user_chat_id
    start_camera = True
    bot.send_message(user_chat_id, 'Камера включена. Поиск человека начался.')
    threading.Thread(target=process_video).start()


def restart(message):
    global start_camera, user_chat_id
    start_camera = True
    bot.send_message(user_chat_id, 'Камера перезапущена. Поиск человека начался.')
    threading.Thread(target=process_video).start()


def route_camera1(message):
    bot.send_message(message.chat.id,
                     'Введите команды для движения:\n/forward [расстояние]\n/backward [расстояние]\n/left [угол]\n/right [угол]\n/save [название маршрута]')


def forward(message):
    distance = message.text.split()[1] if len(message.text.split()) > 1 else ''
    commands = ['forward ' + distance]
    save_route('example_route', commands)
    bot.send_message(message.chat.id, 'Добавлено движение вперед на ' + distance)


def backward(message):
    distance = message.text.split()[1] if len(message.text.split()) > 1 else ''
    commands = ['backward ' + distance]
    save_route('example_route', commands)
    bot.send_message(message.chat.id, 'Добавлено движение назад на ' + distance)


def left(message):
    angle = message.text.split()[1] if len(message.text.split()) > 1 else ''
    commands = ['left ' + angle]
    save_route('example_route', commands)
    bot.send_message(message.chat.id, 'Добавлен поворот налево на ' + angle + ' градусов')


def right(message):
    angle = message.text.split()[1] if len(message.text.split()) > 1 else ''
    commands = ['right ' + angle]
    save_route('example_route', commands)
    bot.send_message(message.chat.id, 'Добавлен поворот направо на ' + angle + ' градусов')


def save_route(route_name, commands):
    with open(os.path.join(routes_folder, route_name + '.txt'), 'w') as f:
        f.write('\n'.join(commands))


bot.message_handler(commands=['start'])(start)
bot.message_handler(commands=['help'])(help)
bot.message_handler(commands=['start_work'])(start_work)
bot.message_handler(commands=['restart'])(restart)
bot.message_handler(commands=['route_camera1'])(route_camera1)
bot.message_handler(commands=['forward'])(forward)
bot.message_handler(commands=['backward'])(backward)
bot.message_handler(commands=['left'])(left)
bot.message_handler(commands=['right'])(right)

threading.Thread(target=bot.polling).start()
