from dataset_config import train_set
import cv2
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.patches import Rectangle

head_motion_data = []  # 사용자의 헤드모션 데이터를 저장할 리스트
eye_motion_data = []
# 파일 경로
file_path = "./HMEM_Data/T00/A380.txt"

# 파일 열기
with open(file_path, 'r') as file:
    # 각 행을 읽어오기
    count=0
    for line in file:
        # 행을 공백을 기준으로 분할하여 리스트로 만들기
        values = line.split()
        if count%2==0:
            # 2번째와 3번째 열의 데이터 가져오기
            if len(values) >= 3:  # 적어도 3개 이상의 열이 있는지 확인
                column_2 = float(values[1])
                column_3 = float(values[2])
                column_5 = float(values[4])
                column_6 = float(values[5])
                
                adjusted_longitude = column_3+180
                adjusted_latitude = column_2 + 90
                x_coordinate = adjusted_longitude
                y_coordinate = adjusted_latitude
                x_pixel = int((x_coordinate / 360) * 3840)
                y_pixel = int((y_coordinate / 180) * 1920)
                
                x_eye=x_pixel-(1080/2)+column_5*1080
                y_eye=y_pixel-(1200/2)+column_6*1200
                # 가져온 데이터를 리스트에 추가
                head_motion_data.append((x_pixel, y_pixel))
                eye_motion_data.append((x_eye,y_eye))
        count+=1

video_folder = "./VIDEO_Data/"
video_info = {}

# 변환을 위한 torchvision.transforms 사용
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 각 비디오에 대한 정보 수집 함수
def get_video_info(video_name):
    video_path = f"{video_folder}/{video_name}.mp4"
    cap = cv2.VideoCapture(video_path)

    # 비디오 정보
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fig, ax = plt.subplots()
    # 비디오의 각 프레임에서 이미지 샘플링
    sample_frames = []
    for i in range(frames):  # 간단한 예제로 30프레임만 샘플링
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sample_frames.append(transform(frame))
            x_head,y_head=head_motion_data[i]
            x_eye,y_eye=eye_motion_data[i]
            ax.clear()  # 이미지 초기화
            ax.imshow(frame)
            
            ax.scatter(x_eye, y_eye, color='blue', s=10) #눈 시점 그리기
            box = Rectangle((x_head - 1080 / 2, y_head - 1200 / 2), 1080, 1200, edgecolor='red', linewidth=2, facecolor='none')#FOV그리기
            ax.add_patch(box)
            
            plt.pause(0.0000000000001)  # 작은 값으로 설정하여 자연스러운 비디오 플레이백
    plt.close()
    cap.release()

    video_info[video_name] = {
        "frames": frames, 
        "fps": fps,
        "width": width,
        "height": height,
        "sample_frames": torch.stack(sample_frames),
    }

# # 각 비디오에 대한 정보 수집
# for video_name in train_set:
#     get_video_info(video_name)
get_video_info(train_set[0])

# 예시로 첫 번째 비디오에 대한 정보 출력
print(video_info[train_set[0]])