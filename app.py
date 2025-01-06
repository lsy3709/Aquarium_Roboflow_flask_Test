# 간단하고 빠르게 웹 애플리케이션을 제작할 수 있는 파이썬 기반 프레임워크
# url_for : 라우팅, send_from_directory: 파일 전송, send_from_directory: JSON 응답,
# render_template 템플릿 렌더링을 지원
from flask import Flask, url_for, send_from_directory, render_template, jsonify
from flask_socketio import SocketIO # 클라이언트와 서버 간 실시간 양방향 통신을 가능
import os
import threading
# 비디오 스트리밍 또는 이미지 처리, 웹캠이나 비디오 파일에서 프레임을 읽고 YOLO 탐지를 적용하는 데 사용
import cv2
# ultralytics: 간단히 YOLO 모델을 로드하고 사용
from ultralytics import YOLO # 실시간 객체 탐지 모델로, 이미지 또는 비디오에서 빠르고 정확하게 객체를 탐지
from flask import Response, request, send_file, abort

# Flask와 SocketIO 초기화
# 현재 모듈의 이름을 나타냅니다. Flask는 이를 사용해 애플리케이션의 경로를 설정, 애플리케이션의 라우트, 설정, 확장 등을 관리
app = Flask(__name__)
# SocketIO 객체를 생성하고 Flask 애플리케이션에 연결
# *은 모든 출처에서의 요청을 허용,
# 다양한 출처에서 클라이언트가 연결할 수 있도록 하는 설정으로, 특히 개발 단계에서 유용
# SocketIO는 클라이언트와 서버 간 실시간 양방향 통신을 가능
socketio = SocketIO(app, cors_allowed_origins="*")

# 결과 저장 폴더 설정
# 사용자가 이미지를 업로드하거나 동영상을 제공할 때, 해당 파일들이 저장되는 디렉토리
# 'uploads'는 프로젝트 루트 디렉토리 아래 생성
UPLOAD_FOLDER = 'uploads'
# 처리 결과(예: YOLO로 탐지된 이미지나 비디오)를 저장할 폴더 경로를 설정.
# 모델이 처리한 후 생성된 파일(탐지 결과 이미지, JSON 데이터 등)을 관리하기 위한 디렉토리
RESULT_FOLDER = 'results'
# 업로드 폴더가 없으면 새로 생성합니다.
# exist_ok=True:
# 폴더가 이미 존재할 경우 에러를 발생시키지 않고 그냥 지나갑니다.
# 이로 인해 코드 실행 시 항상 안정적으로 폴더가 준비
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# 결과 폴더를 생성
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Flask 라우트를 정의
# 클라이언트에서 접근할 수 있는 기본 웹 페이지를 정의
# '/':
# 기본 URL 경로(루트 경로)를 의미합니다. 예: http://127.0.0.1:5000/
# 이 경로에 접속하면 지정된 함수가 실행
@app.route('/')
# 루트 경로(/)에 대한 요청이 들어왔을 때 실행되는 함수입니다.
# 반환값은 클라이언트에 전송됩니다.
# 여기서는 index.html 템플릿 파일을 렌더링해 반환합니다.
def index():
    # Flask의 render_template 함수는 지정된 HTML 템플릿을 렌더링하고 클라이언트로 반환합니다.
    # index.html:
    # 프로젝트의 templates 디렉토리 안에 있어야 합니다. Flask는 기본적으로 이 디렉토리에서 템플릿 파일을 찾습니다.
    # 예:
    # templates/index.html 파일이 클라이언트에 표시됩니다.
    return render_template('index.html')

# YOLO 모델 로드
# best.pt: 사전에 학습된 YOLO 모델의 가중치 파일입니다.
# 이 파일을 사용하여 이미지나 비디오에서 객체 탐지를 수행합니다
model = YOLO("best.pt")
# 127.0.0.1: 로컬 IP 주소(개발 환경에서 사용). = localhost:5000
app.config['SERVER_NAME'] = '127.0.0.1:5000'  # Flask 서버 주소와 포트 설정

# 파일(이미지 또는 비디오)을 YOLO 모델로 처리하는 함수입니다.
# 매개변수:
# file_path: 처리할 원본 파일의 경로.
# output_path: 처리 결과를 저장할 파일의 경로.
# file_type: 파일 유형 ('image' 또는 'video').
# YOLO 모델의 탐지 결과를 이미지 또는 비디오로 저장하며, 비동기로 실행되어 서버의 응답 속도를 높입니다.
def process_file(file_path, output_path, file_type):
    """비동기로 파일 처리"""
    # Flask 애플리케이션 컨텍스트를 활성화합니다.
    # Flask는 애플리케이션의 설정 및 리소스를 사용하는 작업이 있을 때
    # 명시적으로 컨텍스트를 활성화해야 합니다.
    # 이 부분은 Flask 서버 외부에서 실행되는
    # 작업(예: 비동기 작업)에서도 안전하게 Flask의 리소스를 사용할 수 있도록 합니다.
    with app.app_context():  # 애플리케이션 컨텍스트 활성화
        if file_type == 'image':
            # file_path의 이미지를 모델에 입력하여 탐지를 수행합니다.
            results = model(file_path)
            # 탐지된 결과를 프레임에 시각적으로 그립니다.
            result_img = results[0].plot()
            # 처리된 이미지를 output_path에 저장합니다.
            cv2.imwrite(output_path, result_img)
        elif file_type == 'video':
            # OpenCV를 사용해 비디오 파일을 읽습니다.
            cap = cv2.VideoCapture(file_path)
            # 비디오의 가로, 세로 해상도와 초당 프레임 수(FPS)를 가져옵니다.
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            # 탐지 결과를 저장할 비디오 파일을 설정합니다.
            # mp4v'는 MP4 포맷을 지원하는 코덱입니다.
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # 필요에 따라 다른 코덱(H.264 등)을 사용할 수도 있습니다.
            # fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 코덱
            # 비디오 쓰기
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            # 프레임 처리 루프
            # 비디오의 각 프레임을 YOLO 모델로 처리하고 탐지 결과를 저장합니다.
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model(frame)
                result_frame = results[0].plot()
                out.write(result_frame)
            # 리소스 해제
            # 비디오 객체와 파일 쓰기를 종료하고 리소스를 해제합니다.
            cap.release()
            out.release()


        # Flask-SocketIO를 사용해 처리 완료 상태를 클라이언트에 알립니다.
        socketio.emit(
            # 'file_processed':
            # 파일 처리가 완료되었음을 나타내는 이벤트입니다.
            'file_processed',
            # 클라이언트가 결과 파일을 다운로드할 수 있는 URL을 생성하여 전달합니다.
            # url_for('download_file', ...):
            # Flask의 특정 라우트를 기반으로 URL을 생성합니다.
            # _external=True는 절대 URL(예: http://127.0.0.1:5000/...)을 생성합니다.
            {'url': url_for('download_file', filename=os.path.basename(output_path), _external=True)}
        )

# 클라이언트가 처리된 파일(이미지 또는 비디오)을 다운로드할 수 있도록 경로를 설정합니다.
# <filename>:
# 클라이언트 요청 시 다운로드할 파일명을 URL 경로로 전달합니다.
# 예: http://127.0.0.1:5000/download/result_example.mp4
@app.route('/download/<filename>')
# 다운로드 요청을 처리하는 함수입니다.
def download_file(filename):
    """YOLOv8 처리된 동영상 다운로드"""
    file_path = os.path.join(RESULT_FOLDER, filename)
    # 결과 폴더(RESULT_FOLDER)에 요청한 파일이 있는지 확인합니다.
    if not os.path.isfile(file_path):
        # # 파일이 없으면 404 에러와 함께 JSON 응답을 반환합니다.
        return jsonify({"error": "File not found"}), 404
    # Flask의 send_file을 사용해 파일을 클라이언트에게 전송합니다.
    # as_attachment=True:
    # 파일을 첨부 파일로 다운로드하도록 설정합니다.
    # download_name=filename:
    # 다운로드 시 클라이언트가 볼 파일 이름을 지정합니다.
    return send_file(file_path, as_attachment=True, download_name=filename)

# 클라이언트가 파일을 서버로 업로드할 수 있는 경로입니다.
@app.route('/upload', methods=['POST'])
# 업로드된 파일을 처리하는 함수입니다
def upload_file():
    # 클라이언트 요청에 파일 데이터가 포함되어 있는지 확인합니다.
    if 'file' not in request.files:
        # 없을 경우 400 에러를 반환합니다.
        return jsonify({"error": "No file part"}), 400

    # 업로드된 파일이 선택되었는지 확인합니다. 파일 이름이 비어 있다면 에러를 반환합니다.
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # filename = secure_filename(file.filename)
    print("file.filename : " + file.filename)
    # 업로드된 파일의 이름을 가져옵니다.
    filename = file.filename
    # 파일 이름을 사용하여 경로를 구성하므로 파일 이름이 고유해야 충돌이 발생하지 않습니다.
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    # UPLOAD_FOLDER에 파일을 저장합니다.
    file.save(file_path)
    # 처리 결과 파일의 이름을 result_ 접두어를 붙여 설정합니다.
    output_filename = f"result_{filename}"
    # 결과 파일은 RESULT_FOLDER에 저장됩니다.
    output_path = os.path.join(RESULT_FOLDER, output_filename)

    print("filename : " + filename)
    # 업로드된 파일의 확장자를 확인하여 이미지(image)인지 비디오(video)인지 판별합니다.
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_type = 'image'
    elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        file_type = 'video'
    else:
        # 지원하지 않는 파일 형식일 경우 400 에러를 반환합니다.
        return jsonify({"error": "Unsupported file type"}), 400

    # 파일 처리는 시간이 오래 걸릴 수 있으므로 비동기로 실행하여 서버 응답 속도를 높입니다.
    # process_file:
    # 파일 처리 로직이 정의된 함수(위의 코드 참고).
    # YOLO 모델로 이미지 또는 비디오를 처리하고 결과를 저장합니다.
    thread = threading.Thread(target=process_file, args=(file_path, output_path, file_type))
    # 새로운 스레드에서 처리 작업을 시작합니다.
    thread.start()
    # 비동기 처리를 시작한 후 클라이언트에게 "처리 시작" 메시지를 JSON 형식으로 반환합니다.
    return jsonify({"message": "Processing started"})

# YOLO 모델 처리 결과 파일을 클라이언트에게 제공하는 경로입니다.
# <filename>:
# 클라이언트가 요청한 결과 파일의 이름입니다.
# 예: http://127.0.0.1:5000/results/result_image.jpg
@app.route('/results/<filename>')
# 요청된 결과 파일을 제공하는 함수입니다.
def serve_result(filename):
    """결과 파일 제공"""
    # Flask에서 지정된 디렉토리(RESULT_FOLDER)에서 파일을 클라이언트로 전송합니다.
    return send_from_directory(RESULT_FOLDER, filename)

# Flask 애플리케이션의 시작점을 정의합니다.
if __name__ == '__main__':
    # socketio.run(app, debug=True):
    # Flask-SocketIO를 사용해 애플리케이션을 실행합니다.
    # debug=True:
    # 코드 변경 시 자동으로 서버를 재시작하며, 디버그 정보를 출력합니다.
    socketio.run(app, debug=True)
