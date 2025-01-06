from flask import Flask, request, redirect, url_for, send_from_directory, jsonify, render_template
from flask_socketio import SocketIO, emit
import os
import threading
import cv2
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import unicodedata
from flask import send_file

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# 결과 저장 폴더 설정
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# YOLO 모델 로드
model = YOLO("best.pt")
app.config['SERVER_NAME'] = '127.0.0.1:5000'  # Flask 서버 주소와 포트 설정

def normalize_filename(filename):
    """파일 이름을 안전하게 변환 (한글 포함)"""
    # 파일 이름에서 디렉토리 경로 제거
    base_filename = os.path.basename(filename)
    # 한글과 특수 문자를 NFC(normalization)로 정규화
    normalized = unicodedata.normalize('NFC', base_filename)
    # 안전한 파일 이름으로 변환
    return secure_filename(normalized)

def process_file(file_path, output_path, file_type):
    """비동기로 파일 처리"""
    with app.app_context():  # 애플리케이션 컨텍스트 활성화
        if file_type == 'image':
            results = model(file_path)
            result_img = results[0].plot()
            cv2.imwrite(output_path, result_img)
        elif file_type == 'video':
            cap = cv2.VideoCapture(file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                result_frame = results[0].plot()
                out.write(result_frame)

            cap.release()
            out.release()

        # 처리 완료 알림
        socketio.emit(
            'file_processed',
            # {'url': url_for('serve_video', filename=os.path.basename(output_path), _external=True)}
            {'url': url_for('download_file', filename=os.path.basename(output_path), _external=True)}
        )

@app.route('/download/<filename>')
def download_file(filename):
    """YOLOv8 처리된 동영상 다운로드"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.isfile(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(file_path, as_attachment=True, download_name=filename)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # filename = secure_filename(file.filename)
    print("file.filename : " + file.filename)
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    output_filename = f"result_{filename}"
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    print("filename : " + filename)
    # 파일 유형 확인 및 처리
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_type = 'image'
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        file_type = 'video'
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # 비동기로 파일 처리
    thread = threading.Thread(target=process_file, args=(file_path, output_path, file_type))
    thread.start()

    return jsonify({"message": "Processing started"})

@app.route('/results/<filename>')
def serve_result(filename):
    """결과 파일 제공"""
    return send_from_directory(RESULT_FOLDER, filename)

@app.route('/videos/<filename>')
def serve_video(filename):
    """동영상 파일 제공"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    if not os.path.isfile(file_path):
        return os.abort(404, description="File not found")
    return send_file(file_path)

if __name__ == '__main__':
    socketio.run(app, debug=True)
