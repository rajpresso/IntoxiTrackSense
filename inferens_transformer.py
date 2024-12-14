import os
import shutil
import uvicorn
import cv2
import csv
import torch
import gc
from ultralytics import YOLO
import numpy as np
import pandas as pd
import torch.nn as nn
import pickle
import os
import shutil
import subprocess

from insert_log_info import insert_log
from datetime import datetime
from config.config import db_config
from sklearn.preprocessing import StandardScaler


## 데이터 각도 변경 
# 각도 계산을 위한 키포인트 인덱스
keypoint_indices = {
    "right_arm": ("x7","y7", "x9","y9", "x11","y11"),
    "left_arm": ("x6","y6", "x8","y8", "x10","y10"),
    "right_leg": ("x13","y13", "x15","y15","x17","y17"),
    "left_leg":("x12","y12","x14","y14","x16","y16")
}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 모델 설정
model = YOLO('./models/yolov10x/yolov10x.pt')
model.overrides['imgsz'] = 1920

model.to(device)
pose = YOLO('./models/yolov8l-pose/yolov8l-pose.pt')
pose.to(device)


def calculate_angle(pointA, pointB, pointC):
	# 벡터 계산
	BA = np.array(pointA) - np.array(pointB)
	BC = np.array(pointC) - np.array(pointB)
	# 내적과 벡터 크기 계산
	dot_product = np.dot(BA, BC)
	magnitude_BA = np.linalg.norm(BA)
	magnitude_BC = np.linalg.norm(BC)
	# 각도 계산 (라디안 -> 도)
	cos_theta = dot_product / (magnitude_BA * magnitude_BC + 1e-6)  # 0으로 나누는 오류 방지
	angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
	return np.degrees(angle)


def calculate_angles_from_dataframe(data, keypoint_indices):
    angles = []  # 각도를 저장할 리스트   
    # 데이터프레임 순회 및 각도 계산
    for _, row in data.iterrows():
        frame_angles = {}
        for part, (xA, yA, xB, yB, xC, yC) in keypoint_indices.items():
            pointA = (row[xA], row[yA])
            pointB = (row[xB], row[yB])
            pointC = (row[xC], row[yC])
            frame_angles[part] = calculate_angle(pointA, pointB, pointC)
        angles.append(frame_angles)    
    
    # 각도를 데이터프레임으로 변환 및 병합
    angle_df = pd.DataFrame(angles)
    angle_df["frame"] = data['frame']
    angle_df['label']=data['label']
    
    # angle_df.to_csv('/home/alpaco/osh_rsj/final/degreecsv.csv')
    return angle_df

event_logs = {
	"intoxicated_person_appeared": None,
	"intoxicated_person_touched_car": None,
	"intoxicated_person_overlap_open_driver_seat": None,
	"car_started_moving": None
}

def log_event(event,frame_count):
	# 이벤트 로그 갱신
	if event not in event_logs or event_logs[event] is None:
		event_logs[event] = frame_count
		print(f"Event logged: {event} at frame {frame_count}")
	insert_log('1번 모니터',event ,frame_count)

def boxes_overlap(boxA, boxB):
	# 두 바운딩 박스가 겹치는지 확인하는 함수
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	if xA < xB and yA < yB:
		return True
	else:
		return False

def process_video_with_dual_csv(input_video_path, output_video_path, abs_csv_path, rel_csv_path):

	# 비디오 파일 열기
	cap = cv2.VideoCapture(input_video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps == 0:
		print(f"Error: {input_video_path} 비디오 파일을 열 수 없습니다.")
		return
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# 비디오 저장 설정
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

	# 데이터 저장을 위한 딕셔너리 초기화
	data = {}

	# CSV 파일 생성 및 헤더 작성
	with open(abs_csv_path, mode='w', newline='') as abs_csv, open(rel_csv_path, mode='w', newline='') as rel_csv:
		abs_writer = csv.writer(abs_csv)
		rel_writer = csv.writer(rel_csv)

		# 헤더 작성
		header = ['frame']
		for i in range(1, 18):
			header.extend([f'x{i}', f'y{i}'])
		header.append('label')
		abs_writer.writerow(header)
		rel_writer.writerow(header)

		frame_count = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				break

			# 원본 프레임 복사
			results = model.track(frame, persist=True, conf=0.1)
			existing_labels = set()

			# 현재 프레임의 데이터를 저장할 리스트 초기화
			data[frame_count] = []

			for result in results:
				if result.boxes.id is not None:
					for box, track_id, cls_id, conf in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls, result.boxes.conf):
						if int(cls_id) == 0:  # 사람 클래스만 필터링
							x1, y1, x2, y2 = map(int, box)
							label = f'ID: {track_id}'
							if label in existing_labels:
								continue
							existing_labels.add(label)

							# BBox 외부를 검정색으로 마스킹
							masked_frame = frame.copy()
							masked_frame[:y1, :] = 0
							masked_frame[y2:, :] = 0
							masked_frame[:, :x1] = 0
							masked_frame[:, x2:] = 0

							# 현재 BBox에 대해 YOLO Pose 모델 적용
							keypoints_results = pose(masked_frame, imgsz=800)
							if hasattr(keypoints_results[0], 'keypoints'):
								keypoints = keypoints_results[0].keypoints.xy.cpu().numpy()[0]
								if keypoints.shape[0] >= 17:
									# 절대 좌표로 저장 (첫 번째 CSV 파일)
									abs_row_data = [frame_count]
									# 상대 좌표로 저장 (두 번째 CSV 파일)
									rel_row_data = [frame_count]

									for point in keypoints:
										x_kp, y_kp = int(point[0]), int(point[1])
										# 절대 좌표
										abs_row_data.extend([x_kp, y_kp])
										# 상대 좌표 (BBox 기준)
										rel_x_kp = x_kp - x1 if x_kp > 0 else x_kp
										rel_y_kp = y_kp - y1 if y_kp > 0 else y_kp
										rel_row_data.extend([rel_x_kp, rel_y_kp])

									abs_row_data.append(label)
									rel_row_data.append(label)

									# CSV 파일에 작성
									abs_writer.writerow(abs_row_data)
									rel_writer.writerow(rel_row_data)

									# 데이터 저장
									detection = {
										'id': int(track_id),
										'bbox': (x1, y1, x2, y2),
										'keypoints': keypoints,
										'conf': float(conf)
									}
									data[frame_count].append(detection)

			frame_count += 1
			torch.cuda.empty_cache()
			gc.collect()

	cap.release()
	out.release()
	print(f"{output_video_path} 파일로 동영상 저장이 완료되었습니다.")
	print(f"절대 좌표 CSV: {abs_csv_path}")
	print(f"상대 좌표 CSV: {rel_csv_path}")

	# 데이터 저장
	with open('detection_data.pkl', 'wb') as f:
		pickle.dump(data, f)
	print("Detection data saved to detection_data.pkl")


def create_sequences(df, seq_length):
	xs, pid = [], []
	for _, group in df.groupby(['label']):
		if len(group) < 5:
			continue
		print(len(group))
		
		group = group.sort_values(by=['frame']).reset_index(drop=True)

		# 전체 프레임 생성
		all_frames = pd.DataFrame({'frame': np.arange(0, seq_length)})

		# 누락된 프레임을 결합
		group = all_frames.merge(group, on='frame', how='left')
		#group.interpolate(method='linear', inplace=True, axis=0) #,  limit_direction='both')
		#print(group)
		# 결측값이 여전히 남아 있다면 0으로 채우기 (필요 시)
		#print(group)
		group.fillna(0, inplace=True)
		group['label']= group['label'].astype(str)
		
		# 'frame'과 'label' 제외한 데이터로 시퀀스 생성
		temp= group['label'].str.replace("ID: ", "").dropna().unique().tolist()
		temp.remove("0")
		# 'frame'과 'label' 제외한 데이터로 시퀀스 생성
		data_X = group.drop(columns=['frame', 'label'], errors='ignore').values
		xs.append(data_X)

		pid.append(int(float(temp[0])))
	return np.array(xs), pid


# Transformer 모델을 위한 설정
class TransformerModel(torch.nn.Module):
	def __init__(self, input_size, hidden_size, num_classes, num_heads=2, num_layers=4, dropout=0.1):
		super(TransformerModel, self).__init__()
		
		# Multi-Head Attention 레이어
		self.attention = torch.nn.MultiheadAttention(embed_dim=input_size, num_heads=num_heads, dropout=dropout)
		
		# Transformer Encoder
		self.transformer = torch.nn.TransformerEncoder(
			torch.nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout), 
			num_layers=num_layers
		)
		
		# Fully connected layers
		self.fc = torch.nn.Sequential(
			torch.nn.Linear(input_size, hidden_size),
			torch.nn.ReLU(),
			torch.nn.Dropout(dropout),
			torch.nn.Linear(hidden_size, num_classes)
		)

	def forward(self, x):
		# 시퀀스 길이, 배치 크기, 특성 차원에 맞게 변환
		x = x.transpose(0, 1)  # Transformer는 (seq_len, batch_size, features)의 형태를 기대함
		# Attention 통과
		attn_output, _ = self.attention(x, x, x)
		# Transformer Encoder 통과
		transformer_output = self.transformer(attn_output)
		# 마지막 시퀀스 출력을 사용 (기본적으로 클래스 레이블 예측)
		output = transformer_output[-1, :, :]
		# Fully connected layers 통과
		output = self.fc(output)
		
		return output


def process_video_with_overlay(input_video_path, output_video_path, detection_data_path, suspect_ids, vehicle_model_path):
	# 차량 및 차문 검출 모델 로드
	vehicle_model = YOLO(vehicle_model_path)
	vehicle_model.to(device)
	
	# 차량 및 차문 클래스 이름과 ID 확인
	vehicle_class_names = vehicle_model.model.names
	print("Vehicle Model Classes:", vehicle_class_names)
	# 예시: {0: 'car', 1: 'bus', 2: 'truck', 3: 'open_driver_seat'}
	
	# 주취자 검출 데이터 로드
	with open(detection_data_path, 'rb') as f:
		person_data = pickle.load(f)
	
	# 차량 바운딩 박스를 저장할 딕셔너리 초기화
	car_tracks = {}
	door_tracks = {}
	
	# 주취자가 접촉한 차량의 정보를 저장할 딕셔너리
	suspect_car_info = {}
	
	# 비디오 파일 열기
	cap = cv2.VideoCapture(input_video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	if fps == 0:
		print(f"Error: {input_video_path} 비디오 파일을 열 수 없습니다.")
		return
	frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	# 비디오 저장 설정
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

	frame_count = 0

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# 현재 프레임의 주취자 데이터 가져오기
		person_detections = person_data.get(frame_count, [])

		# 차량 및 차문 검출 및 추적
		vehicle_results = vehicle_model.track(frame, conf=0.5, persist=True, imgsz=1024)
		current_car_ids = set()
		current_car_centers = {}  # 현재 프레임의 차량 중심 좌표
		current_door_bboxes = []  # 현재 프레임의 모든 open_driver_seat 바운딩 박스 리스트 (conf >= 0.7인 것만)

		for result in vehicle_results:
			if result.boxes.id is not None:
				for box, cls_id, track_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.id, result.boxes.conf):
					x1, y1, x2, y2 = map(int, box)
					cls_id = int(cls_id)
					track_id = int(track_id)
					conf = float(conf)
					class_name = vehicle_class_names[cls_id]
					
					# 바운딩 박스 그리기 및 라벨 표시
					if cls_id == 0:  # 자동차
						car_tracks[track_id] = {'bbox': (x1, y1, x2, y2), 'last_seen': frame_count}
						current_car_ids.add(track_id)
						# 현재 차량의 중심 좌표 계산
						center_x = (x1 + x2) / 2
						center_y = (y1 + y2) / 2
						current_car_centers[track_id] = (center_x, center_y)
						
						color = (255, 0, 0)  # 파란색 - 자동차
						# 주취자가 접촉한 차량인지 확인
						if track_id in suspect_car_info:
							label = f'Intoxicated person vehicle {class_name} ID:{track_id} conf:{conf:.2f}'
						else:
							label = f'{class_name} ID:{track_id} conf:{conf:.2f}'
						cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
						cv2.putText(frame, label, (x1, y1 - 10),
									cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
					elif cls_id == 3 and conf >= 0.7:  # 차문 (open_driver_seat) & conf >= 0.7
						door_tracks[track_id] = {'bbox': (x1, y1, x2, y2), 'last_seen': frame_count}
						current_door_bboxes.append((x1, y1, x2, y2))
						
						color = (0, 255, 255)  # 노란색 - open_driver_seat
						label = f'{class_name} ID:{track_id} conf:{conf:.2f}'
						cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
						cv2.putText(frame, label, (x1, y1 - 10),
									cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
					else:
						continue  # 기타 클래스는 무시하거나 conf < 0.7인 open_driver_seat 무시

		# 오버레이 상태 초기화
		overlay_color = None
		overlay_priority = 0  # 우선순위: 높을수록 높은 우선순위

		for person_det in person_detections:
			track_id = person_det['id']
			person_bbox = person_det['bbox']
			keypoints = person_det['keypoints']
			conf = person_det['conf']

			x1_p, y1_p, x2_p, y2_p = person_bbox
			label = f'ID: {track_id} conf: {conf:.2f}'

			# 주취자 여부에 따라 색상 설정
			if track_id in suspect_ids:
				color = (0, 0, 255)  # 빨간색
			else:
				color = (0, 255, 0)  # 초록색

			# 주취자 바운딩 박스 및 키포인트 그리기
			cv2.rectangle(frame, (x1_p, y1_p), (x2_p, y2_p), color, 2)
			cv2.putText(frame, label, (x1_p, y1_p - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

			for point in keypoints:
				x, y = int(point[0]), int(point[1])
				cv2.circle(frame, (x, y), radius=5, color=color, thickness=-1)

			# 주취자인 경우에만 처리
			if track_id in suspect_ids:
				# 우선순위 1: 주취자 등장 - 노란색 (이 부분은 주석 처리되어 있음)
				# if overlay_priority < 1:
				#     overlay_color = (0, 255, 255)  # 노란색 (BGR)
				#     overlay_priority = 1
				if event_logs["intoxicated_person_appeared"] is None:
					log_event("intoxicated_person_appeared",frame_count)
				# 차량과의 겹침 확인
				for car_id, car_info in car_tracks.items():
					car_bbox = car_info['bbox']
					if boxes_overlap(person_bbox, car_bbox):
						# 우선순위 2: 주취자가 차량과 겹침 - 주황색
						if overlay_priority < 2:
							if event_logs["intoxicated_person_touched_car"] is None:
								log_event('intoxicated_person_touched_car' ,frame_count)
							overlay_color = (0, 165, 255)  # 주황색 (BGR)
							overlay_priority = 2
						# 주취자가 접촉한 차량 정보 저장
						if car_id not in suspect_car_info:
							# 최초 접촉 시 차량의 초기 중심 좌표 저장
							initial_center = current_car_centers.get(car_id)
							suspect_car_info[car_id] = {'initial_center': initial_center, 'person_bbox': person_bbox}
							print(f"Vehicle ID {car_id} initial center recorded: {initial_center}")
						else:
							suspect_car_info[car_id]['person_bbox'] = person_bbox
						break  # 겹치는 차량이 있으면 더 이상 확인하지 않음

		# 주취자가 접촉한 차량의 바운딩 박스와 현재 프레임의 모든 open_driver_seat 바운딩 박스 간의 3중 겹침 확인
		for car_id in suspect_car_info.keys():
			car_bbox = car_tracks.get(car_id, {}).get('bbox')
			if car_bbox is None:
				continue
			person_bbox = suspect_car_info[car_id].get('person_bbox')
			if person_bbox is None:
				continue
			# 우선순위 3: 주취자, 차량, 차문이 모두 겹칠 때 - 핑크색
			if overlay_priority < 3:
				for door_bbox in current_door_bboxes:
					if boxes_overlap(car_bbox, door_bbox) and boxes_overlap(person_bbox, door_bbox):
						overlay_color = (147, 20, 255)  # 핑크색 (BGR)
						if event_logs["intoxicated_person_overlap_open_driver_seat"] is None:
							log_event('intoxicated_person_overlap_open_driver_seat' ,frame_count)
						overlay_priority = 3
						break
			if overlay_priority == 3:
				continue  # 핑크색 오버레이가 설정되면 다음 차량으로

			# 우선순위 4: 차량 중심 이동이 200픽셀 이상 - 빨간색
			# 현재 프레임에서 해당 차량이 검출되었는지 확인
			if car_id in current_car_centers and suspect_car_info[car_id]['initial_center'] is not None:
				current_center = current_car_centers[car_id]
				initial_center = suspect_car_info[car_id]['initial_center']
				# 이동 거리 계산
				movement = np.sqrt((current_center[0] - initial_center[0]) ** 2 + (current_center[1] - initial_center[1]) ** 2)
				print(f"Vehicle ID {car_id} movement: {movement}")
				if movement > 200:
					if movement > 200 and event_logs["car_started_moving"] is None:
						log_event('car_started_moving' ,frame_count)
					if overlay_priority < 4:
						overlay_color = (0, 0, 255)  # 빨간색 (BGR)
						overlay_priority = 4
					break  # 빨간색 오버레이가 설정되면 더 이상 확인하지 않음
			else:
				# 차량이 현재 프레임에서 검출되지 않으면 넘어감
				continue

		# 오버레이 적용
		if overlay_color is not None:
			overlay = frame.copy()
			color_overlay = np.full_like(frame, overlay_color, dtype=np.uint8)
			cv2.addWeighted(color_overlay, 0.3, frame, 0.7, 0, frame)

		out.write(frame)
		frame_count += 1

	cap.release()
	out.release()
	print(f"Processed video saved to {output_video_path}")


def infer(video_path):
    print(f'START inferens_transformer {video_path}')
    # 이벤트 로그 초기화
    output_dir = './final/converted'
    output = "./final/test.mp4"  
    abscsv_path = "./final/abscsv.csv"
    relcsv_path = "./final/relcsv.csv"
    os.makedirs(output_dir, exist_ok=True)

    process_video_with_dual_csv(video_path,output,abscsv_path,relcsv_path)
    
    test_data= pd.read_csv("./final/abscsv.csv")
    print(f'test_data shape : {test_data.shape}')
    angle_df = calculate_angles_from_dataframe(test_data, keypoint_indices)
    print(f'calculate_angles_from_dataframe angle_df shape : {angle_df.shape}')

    coordinate_cols = ['right_arm','left_arm','right_leg','left_leg']
    X = angle_df[coordinate_cols].values 

    scaler_X = StandardScaler()
    X_normalized = scaler_X.fit_transform(X)
    angle_df[coordinate_cols] = X_normalized

    sequence_length = 90
    # 시퀀스 생성
    X_seq, pid= create_sequences(angle_df, sequence_length)
    print("end create_sequences")
    test_X_tensor = torch.FloatTensor(X_seq)

    # 모델 초기화
    input_size = X_seq.shape[2]
    hidden_size = 50
    num_layers = 1
    loaded_model = TransformerModel(input_size,hidden_size,num_layers)
    loaded_model.load_state_dict(torch.load('/home/alpaco/project/drunk_prj/models/only_model/1205_onlydegree2.pt'))
    print(" TransformerModel")
    loaded_model.to(device)
    loaded_model.eval()
    suspect_ids = []

    for i in range(len(test_X_tensor)):
        data = test_X_tensor[i]
        track_id = pid[i]
        inputs = data.unsqueeze(0).to(device)
        # 모델 예측
        outputs = loaded_model(inputs)
        pred_prob = torch.sigmoid(outputs).item()
        preds = pred_prob > 0.5  # 이진 분류로 변환
        print(f'ID: {track_id}, Probability: {pred_prob}')
        if preds:
            suspect_ids.append(track_id)
    print("Suspect IDs:", suspect_ids)

    # 차량 및 차문 검출 모델 경로
    vehicle_model_path = './models/result_model/vehicle_model/experiment241125_1024_final/weights/best.pt'

    # 오버레이 적용 함수 호출
    process_video_with_overlay(
        input_video_path=video_path,
        output_video_path='./final/overlay_video.mp4',
        detection_data_path='detection_data.pkl',
        suspect_ids=suspect_ids,
        vehicle_model_path=vehicle_model_path
    )
    return event_logs  # 최종적으로 이벤트 로그를 반환

