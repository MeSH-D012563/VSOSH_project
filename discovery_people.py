import cv2
import torch
import numpy as np
import threading
import telebot

# Загружаем YOLOv5s
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Загружаем имена классов
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

camera_index = 0

# Флаг для управления камерой
start_camera = False

# Чат id пользователя с которым сейчас работаем
user_chat_id = None

# Функция для обработки видео
def process_video():
    global start_camera, user_chat_id
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть камеру с индексом {camera_index}")
        return

    frame_count = 0
    while start_camera:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр")
            break

        frame_count += 1
        if frame_count % 3 == 0:
            frame = cv2.resize(frame, (320, 240))
            height, width, channels = frame.shape
            results = model(frame)
            class_ids = []
            confidences = []
            boxes = []

            for result in results.xyxy[0]:
                x1, y1, x2, y2, confidence, class_id = result
                if confidence > 0.5 and classes[int(class_id)] == "person":
                    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            mask = np.zeros((height, width), dtype=np.uint8)
            person_detected = False

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    person_detected = True

            if person_detected:
                print("Человек обнаружен!")
                cv2.imwrite('detected_person.jpg', frame)
                start_camera = False
                if user_chat_id:
                    bot.send_photo(user_chat_id, photo=open('detected_person.jpg', 'rb'))
                break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Функция для обработки команды /start
def start(message):
    global start_camera, user_chat_id
    user_chat_id = message.chat.id
    start_camera = True
    bot.send_message(user_chat_id, 'Камера включена. Поиск человека начался.')
    threading.Thread(target=process_video).start()

# Создаем экземпляр бота
bot = telebot.TeleBot("YOUR_BOT_TOKEN")

# Обработчик команды /start
bot.message_handler(commands=['start'])(start)

# Запуск бота
bot.polling()
