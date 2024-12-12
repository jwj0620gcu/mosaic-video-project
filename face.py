import cv2

# 1. 영상 파일 경로 입력
video_path = input("모자이크할 영상 경로를 입력하세요: ")
output_path = "mosaic_output.mp4"

# 2. 영상 열기
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"영상을 열 수 없습니다: {video_path}. 경로를 확인하세요.")
    exit()

# 3. 영상 정보 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# 4. 얼굴 검출을 위한 Haar Cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_cascade.empty():
    print("Haar Cascade를 로드할 수 없습니다.")
    exit()

# 5. 프레임 처리
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("더 이상 프레임을 읽을 수 없습니다. 영상 끝.")
        break
    
    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # 얼굴에 모자이크 처리
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (w // 10, h // 10))  # 축소
        face = cv2.resize(face, (w, h), interpolation=cv2.INTER_NEAREST)  # 확대
        frame[y:y+h, x:x+w] = face

    # 처리된 프레임 저장
    out.write(frame)

# 6. 리소스 해제
cap.release()
out.release()
print(f"모자이크된 영상이 '{output_path}'로 저장되었습니다.")
