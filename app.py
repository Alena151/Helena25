from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import os
from datetime import datetime
import sqlite3
from ultralytics import YOLO
from flask_cors import CORS 
import base64

model = YOLO("yolov8n.pt")

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'result'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------- Инициализация базы данных ----------------

def init_db():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            object_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()


# ---------------- Главная страница ----------------

@app.route('/')
def index():
    return render_template('index.html')


# ---------------- Обработка изображения ----------------

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    filename = file.filename.lower()

    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if img is None:
        print("Ошибка декодирования изображения")
        return jsonify({"error": "Невозможно прочитать изображение"}), 400

    # YOLOv8 обработка
    results = model.predict(img)[0]
    horse_boxes = []
    for result in results:
        if result.boxes.cls.cpu().numpy() == 17: # ищем объект с классом 17: лошади
            h_box = result.boxes.xyxy[0].numpy().astype(int) # сохраняем координаты мотоцикла
            horse_boxes.append(list(h_box)) # добавляем данные в массив, тут будет список всех найденных мотоциклов

    for box in horse_boxes:
            color = (0, 0, 255)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)

    output_path = os.path.join(UPLOAD_FOLDER, 'result.jpg')
    cv2.imwrite(output_path, img)

    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (timestamp, filename, object_count)
        VALUES (?, ?, ?)
    ''', (datetime.now().isoformat(), filename, len(results[0].boxes)))
    conn.commit()
    conn.close()

    print(f"Объектов найдено: {len(horse_boxes)}")
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify(count=len(horse_boxes), processed_image=img_base64)


# ---------------- Точка входа ----------------

if __name__ == '__main__':
    app.run(debug=True)
