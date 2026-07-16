from unittest import result

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from collections import deque
from collections import Counter

import time
try:
    from .GestureClassifier.predict import GesturePredictor
except ImportError:
    from GestureClassifier.predict import GesturePredictor
    
class gestureDetector:
    def __init__(self,
                 camera_id=0,
                 num_hands=2,
                 min_hand_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """
        使用 MediaPipe Tasks API 的手部追踪器
        """
        self.num_hands = num_hands
        self.min_hand_detection_confidence = min_hand_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.predictor_number = GesturePredictor(model_type="mlp")
        self.predictor_gesture = GesturePredictor(model_type="svm", model_name="svm_model_gesture")


        # 创建手部检测器
        base_options = python.BaseOptions(
            model_asset_path='Gesture_Detect\hand_landmarker.task'
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )

        self.detector = vision.HandLandmarker.create_from_options(options)

        # 定义手指关键点索引
        self.finger_tips = [4, 8, 12, 16, 20]  # 指尖
        self.finger_pips = [3, 6, 10, 14, 18]  # 第二关节
        self.finger_names = ['拇指', '食指', '中指', '无名指', '小指']

        # 颜色定义
        self.colors = {
            'connections': (0, 255, 0),
            'landmarks': (0, 0, 255),
            'fingertips': (255, 0, 0),
            'gesture_text': (0, 255, 255)
        }
        self.cap = cv2.VideoCapture(camera_id)
        # 用于平滑关键点
        self.smooth_landmarks = {}
        # EMA参数
        self.alpha = 0.3

        # ===== 新增：手势检测相关变量 =====
        # 存储最近的手部中心点位置（用于检测移动方向）
        self.hand_positions_history = {}
        self.history_length = 4  # 保存最近4帧的位置

        # 手势检测参数
        self.gesture_cooldown = 0.5  # 手势触发冷却时间（秒）
        self.last_gesture_time = 0
        self.current_gesture = "静止"
        self.wait_for_reset = False

        # 移动阈值（像素）
        self.move_threshold = 15
        #速度阈值（像素/秒）
        self.velocity_threshold = 150
        self.reset_velocity = 100          # 恢复阈值


        # 用于平滑检测的队列
        self.gesture_history = deque(maxlen=5)
        # 用于预测投票
        self.number_prediction_history = []
        self.gesture_prediction_history = []

        self.vote_size = 10
        self.lastest_number=None
        self.lastest_gesture=None



        print("手势识别已初始化！")
        print("支持手势: 左滑, 右滑, 上滑, 下滑, 前推, 后拉")



    def detect_hands(self, frame):
        """检测手部"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)
        smoothed_hands = {}

        for hand_id, hand_landmarks in enumerate(detection_result.hand_landmarks):
            handedness = detection_result.handedness[hand_id][0].category_name

            points = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32
            )

            points = self.smooth_hand_landmarks(
                handedness,
                points
            )

            smoothed_hands[handedness] = points

        return frame, detection_result, smoothed_hands
    def smooth_hand_landmarks(self, handedness, landmarks):
        """
        landmarks: numpy数组 (21,3)
        """

        if handedness not in self.smooth_landmarks:
            self.smooth_landmarks[handedness] = landmarks.copy()
        else:
            self.smooth_landmarks[handedness] = (
                self.alpha * landmarks +
                (1 - self.alpha) * self.smooth_landmarks[handedness]
            )

        return self.smooth_landmarks[handedness]

    def draw_hand_landmarks(self, frame, detection_result, smooth_landmarks, draw_connections=True,
                            draw_landmarks=True, draw_indices=False):
        """绘制手部关键点和连线"""
        if smooth_landmarks:
            for handedness, hand_landmarks in smooth_landmarks.items():
                h, w, _ = frame.shape

                points = []
                for landmark in hand_landmarks:
                    x = int(landmark[0] * w)
                    y = int(landmark[1] * h)
                    points.append((x, y))

                if draw_connections:
                    connections = [
                        (0, 1), (1, 2), (2, 3), (3, 4),
                        (0, 5), (5, 6), (6, 7), (7, 8),
                        (0, 9), (9, 10), (10, 11), (11, 12),
                        (0, 13), (13, 14), (14, 15), (15, 16),
                        (0, 17), (17, 18), (18, 19), (19, 20),
                        (5, 9), (9, 13), (13, 17)
                    ]
                    for connection in connections:
                        if connection[0] < len(points) and connection[1] < len(points):
                            cv2.line(frame, points[connection[0]],
                                     points[connection[1]],
                                     self.colors['connections'], 2)

                if draw_landmarks:
                    for idx, point in enumerate(points):
                        color = self.colors['fingertips'] if idx in self.finger_tips else self.colors['landmarks']
                        radius = 8 if idx in self.finger_tips else 4
                        cv2.circle(frame, point, radius, color, -1)

                        if draw_indices:
                            cv2.putText(frame, str(idx), (point[0] - 10, point[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def count_fingers(self, detection_result):
        """计算伸出的手指数量"""
        finger_counts = []
        finger_statuses = []

        if detection_result.hand_landmarks:
            hand_labels = []
            if hasattr(detection_result, 'handedness'):
                for handedness in detection_result.handedness:
                    hand_labels.append(handedness[0].category_name)
            else:
                hand_labels = ['Right'] * len(detection_result.hand_landmarks)

            for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
                fingers = []
                status = []
                landmarks = hand_landmarks
                hand_label = hand_labels[idx] if idx < len(hand_labels) else 'Right'

                thumb_tip = landmarks[self.finger_tips[0]]
                thumb_pip = landmarks[self.finger_pips[0]]

                if hand_label == 'Right':
                    if thumb_tip.x > thumb_pip.x:
                        fingers.append(1)
                        status.append('伸出')
                    else:
                        fingers.append(0)
                        status.append('弯曲')
                else:
                    if thumb_tip.x < thumb_pip.x:
                        fingers.append(1)
                        status.append('伸出')
                    else:
                        fingers.append(0)
                        status.append('弯曲')

                for i in range(1, 5):
                    tip = landmarks[self.finger_tips[i]]
                    pip = landmarks[self.finger_pips[i]]
                    if tip.y < pip.y:
                        fingers.append(1)
                        status.append('伸出')
                    else:
                        fingers.append(0)
                        status.append('弯曲')

                finger_counts.append(sum(fingers))
                finger_statuses.append(status)

        return finger_counts, finger_statuses

    def get_hand_center(self, detection_result, frame_shape):
        """
        获取左右手中心。

        Returns
        -------
        left_center : (x, y) | None
        right_center : (x, y) | None
        """

        h, w = frame_shape[:2]

        left_center = None
        right_center = None

        if not detection_result.hand_landmarks:
            return left_center, right_center

        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):

            wrist = hand_landmarks[0]
            index_base = hand_landmarks[5]

            center_x = int((wrist.x + index_base.x) * 0.5 * w)
            center_y = int((wrist.y + index_base.y) * 0.5 * h)

            handedness = "Right"
            if hasattr(detection_result, "handedness"):
                handedness = detection_result.handedness[i][0].category_name

            if handedness == "Left":
                left_center = (center_x, center_y)
            else:
                right_center = (center_x, center_y)

        return left_center, right_center

    # 手势检测
    def detect_gesture(self, detection_result, frame_shape):
        """
        检测手势方向
        返回: (手势名称, 手势方向)
        方向: 'left', 'right', 'up', 'down', 'forward', 'backward', 'none'
        """
        if not detection_result.hand_landmarks:
            return '无手势', 'none'

        left_center, right_center = self.get_hand_center(detection_result, frame_shape)
        if not left_center and not right_center:
            return '无手势', 'none'

        # 优先使用左手中心，如果没有则使用右手中心
        center = left_center if left_center else right_center
        hand_id = 0

        # 更新位置历史
        if hand_id not in self.hand_positions_history:
            self.hand_positions_history[hand_id] = deque(maxlen=self.history_length)

        # 同时保存手的大小（用于检测前推/后拉）
        if 'hand_size_history' not in self.__dict__:
            self.hand_size_history = {}
        if hand_id not in self.hand_size_history:
            self.hand_size_history[hand_id] = deque(maxlen=self.history_length)

        self.hand_positions_history[hand_id].append({"pos": center,"time": time.time()})

        # # 计算手的大小（手腕到中指指尖的距离）
        # if detection_result.hand_landmarks:
        #     hand_landmarks = detection_result.hand_landmarks[0]
        #     wrist = hand_landmarks[0]
        #     middle_tip = hand_landmarks[12]  # 中指指尖
        #     h, w = frame_shape[:2]

        #     # 计算欧几里得距离
        #     dx = (middle_tip.x - wrist.x) * w
        #     dy = (middle_tip.y - wrist.y) * h
        #     hand_size = np.sqrt(dx * dx + dy * dy)
        #     self.hand_size_history[hand_id].append(hand_size)

        history = self.hand_positions_history[hand_id]

        if len(history) < 4:
            return "检测中...", "none"

        # -------------------------
        # 计算最近3帧速度
        # -------------------------

        velocities = []

        for i in range(len(history)-3, len(history)):
            p1 = history[i-1]
            p2 = history[i]

            dx = p2["pos"][0] - p1["pos"][0]
            dy = p2["pos"][1] - p1["pos"][1]

            dt = p2["time"] - p1["time"]

            if dt <= 0:
                continue

            distance = np.sqrt(dx*dx + dy*dy)

            velocity = distance / dt

            velocities.append({
                "velocity": velocity,
                "dx": dx,
                "dy": dy
            })

        if len(velocities) == 0:
            return "检测中...", "none"

        # -------------------------
        # 最近3帧平均速度
        # -------------------------

        avg_velocity = sum(v["velocity"] for v in velocities) / len(velocities)

        avg_dx = sum(v["dx"] for v in velocities)

        avg_dy = sum(v["dy"] for v in velocities)

        #print(f"velocity={avg_velocity:.1f}")

        # -------------------------
        # 已经触发过
        # -------------------------

        if self.wait_for_reset:

            if avg_velocity < self.reset_velocity:
                self.wait_for_reset = False

            return "等待恢复", "none"

        # -------------------------
        # 速度不够/移动距离不够
        # -------------------------

        if avg_velocity < self.velocity_threshold:
            return "移动过慢", "none"
        if distance < self.move_threshold:
            return "移动距离过短", "none"

        # -------------------------
        # 判断方向
        # -------------------------

        if abs(avg_dx) > abs(avg_dy):

            if avg_dx > 0:
                direction = "right"
                gesture_name = "向右滑动"
            else:
                direction = "left"
                gesture_name = "向左滑动"

        else:

            if avg_dy > 0:
                direction = "down"
                gesture_name = "向下滑动"
            else:
                direction = "up"
                gesture_name = "向上滑动"

        # -------------------------
        # 锁住，等待恢复
        # -------------------------

        self.wait_for_reset = True

        return gesture_name, direction
    
    def classify_hands(self, detection_result, smoothed_hands=None, predictor=None, prediction_history=None):
        if predictor is None:
            predictor = self.predictor_number
        if prediction_history is None:
            prediction_history = self.number_prediction_history if predictor == self.predictor_number else self.gesture_prediction_history
        
        frame_predictions = []

        frame_predictions.append({
                "handedness": "Right",
                "gesture": "Unknown",
                "confidence": None
            })
        frame_predictions.append({
                "handedness": "Left",
                "gesture": "Unknown",
                "confidence": None
            })
        
        # if not detection_result.hand_landmarks:
        #     return None


        for i, hand_landmarks in enumerate(detection_result.hand_landmarks):

            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                dtype=np.float32
            )

            handedness = "Right"

            if hasattr(detection_result, "handedness"):
                handedness = detection_result.handedness[i][0].category_name
            
            if smoothed_hands is not None and handedness in smoothed_hands:
                landmarks = smoothed_hands[handedness]

            gesture, confidence = predictor.predict(
                {
                    "landmarks": landmarks,
                    "handedness": handedness
                }
            )

            for pred in frame_predictions:
                if pred["handedness"] == handedness:
                    pred["gesture"] = gesture
                    pred["confidence"] = confidence
                    break

        # # 保存这一帧
        prediction_history.append(frame_predictions)

        # 不足10帧
        if len(prediction_history) < self.vote_size:
            return None

        result = self.vote_predictions(prediction_history)
        prediction_history.pop(0)        # print("投票结果:", result)
        return result
    

    def vote_predictions(self, history):

        left_votes = []
        right_votes = []

        left_conf = []
        right_conf = []

        for frame in history:

            for pred in frame:

                if pred["handedness"] == "Left":
                    left_votes.append(pred["gesture"])
                    left_conf.append(pred["confidence"])

                elif pred["handedness"] == "Right":
                    right_votes.append(pred["gesture"])
                    right_conf.append(pred["confidence"])

        result = []

        # ---------- Left ----------
        if left_votes:

            gesture = Counter(left_votes).most_common(1)[0][0]

            if gesture == "Unknown":
                confidence = None
            else:
                valid_conf = [
                    c for g, c in zip(left_votes, left_conf)
                    if g == gesture and c is not None
                ]

                confidence = np.mean(valid_conf) if valid_conf else None

            result.append({
                "handedness": "Left",
                "gesture": gesture,
                "confidence": confidence
            })

        # ---------- Right ----------
        if right_votes:

            gesture = Counter(right_votes).most_common(1)[0][0]

            if gesture == "Unknown":
                confidence = None
            else:
                valid_conf = [
                    c for g, c in zip(right_votes, right_conf)
                    if g == gesture and c is not None
                ]

                confidence = np.mean(valid_conf) if valid_conf else None

            result.append({
                "handedness": "Right",
                "gesture": gesture,
                "confidence": confidence
            })

        return result

    def draw_gesture_info(self, frame, detection_result, frame_shape):
        """在画面上显示手势信息"""
        gesture_name, direction = self.detect_gesture(detection_result, frame_shape)

        # 在画面中央显示手势
        h, w = frame.shape[:2]

        # 背景框
        cv2.rectangle(frame, (w // 2 - 150, 10), (w // 2 + 150, 60), (0, 0, 0), -1)
        cv2.rectangle(frame, (w // 2 - 150, 10), (w // 2 + 150, 60), (255, 255, 255), 2)

        # 显示手势文字
        cv2.putText(frame, f"手势: {gesture_name}",
                    (w // 2 - 120, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 如果有手势方向，显示方向箭头
        if direction != 'none':
            arrow_start = (w // 2, 100)
            arrow_length = 50

            if direction == 'left':
                arrow_end = (arrow_start[0] - arrow_length, arrow_start[1])
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
            elif direction == 'right':
                arrow_end = (arrow_start[0] + arrow_length, arrow_start[1])
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
            elif direction == 'up':
                arrow_end = (arrow_start[0], arrow_start[1] - arrow_length)
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)
            elif direction == 'down':
                arrow_end = (arrow_start[0], arrow_start[1] + arrow_length)
                cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 3, tipLength=0.3)

        return direction

    def get_finger_positions(self, detection_result, frame_shape):
        """获取所有关键点的像素坐标"""
        positions = []
        h, w = frame_shape[:2]

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                hand_positions = []
                for landmark in hand_landmarks:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    z = landmark.z
                    hand_positions.append((x, y, z))
                positions.append(hand_positions)

        return positions

    def draw_finger_info(self, frame, detection_result, finger_counts, finger_statuses, predictions=None):
        """显示手指信息"""
        if not detection_result.hand_landmarks:
            return

        # for idx in range(len(detection_result.hand_landmarks)):
        #     x_pos = 10
        #     y_pos = 80 + idx * 120  # 下移位置，给手势显示留空间

        #     count = finger_counts[idx] if idx < len(finger_counts) else 0
        #     cv2.putText(frame, f"手 {idx + 1} 伸出: {count} 根手指",
        #                 (x_pos, y_pos),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        #     if idx < len(finger_statuses):
        #         status = finger_statuses[idx]
        #         for i, (name, state) in enumerate(zip(self.finger_names, status)):
        #             color = (0, 255, 0) if state == '伸出' else (0, 0, 255)
        #             cv2.putText(frame, f"{name}: {state}",
        #                         (x_pos + 10, y_pos + 25 + i * 20),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if predictions:
            self.lastest = predictions  # 获取最新的预测结果
        if self.lastest :
            for i, pred in enumerate(self.lastest):
                cv2.putText(
                    frame,
                    f'{pred["handedness"]}: {pred["gesture"]} ({pred["confidence"]})',
                    (20, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )
    def draw_predictions(self, frame, number_predictions=None, gesture_predictions=None):
        if number_predictions:
            self.lastest_number = number_predictions  # 获取最新的数字预测结果
        if gesture_predictions:
            self.lastest_gesture = gesture_predictions  # 获取最新的手势预测结果
        gesture_map = {
            "unknown": "unknown",
            0:"fist",
            1:"thumbs_up",
            2:"victory",
            3:"rock",
            4:"thumbs_down",
            5:"palm"
            }
        # 显示数字预测结果
        if self.lastest_number:
            for i, pred in enumerate(self.lastest_number):
                cv2.putText(
                    frame,
                    f'{pred["handedness"]}: {pred["gesture"]} ({pred["confidence"]})',
                    (20, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )

        # 显示手势预测结果
        if self.lastest_gesture:
            for i, pred in enumerate(self.lastest_gesture):
                cv2.putText(
                    frame,
                    f'{pred["handedness"]}: {gesture_map.get(pred["gesture"], "Unknown")} ({pred["confidence"]})',
                    (20, 150 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0,255,255),
                    2
                )

    def process_frame(self):
        """处理单帧图像"""
        ret, frame = self.cap.read()
        if not ret:
            print("无法获取摄像头画面")
            return None, None, None

        frame = cv2.flip(frame, 1)  # 翻转图像
        annotated_frame, detection_result, smoothed_hands = self.detect_hands(frame)
        finger_counts, finger_statuses = self.count_fingers(detection_result)
        hand_count = len(detection_result.hand_landmarks)
        positions = self.get_finger_positions(detection_result, frame.shape)

        self.draw_hand_landmarks(annotated_frame, detection_result, smoothed_hands)
        number_predictions = self.classify_hands(detection_result, smoothed_hands=smoothed_hands, predictor=self.predictor_number)
        gesture_predictions = self.classify_hands(detection_result, smoothed_hands=smoothed_hands, predictor=self.predictor_gesture)


        # if number_predictions:
        #     print("数字预测结果:", number_predictions)
        # if gesture_predictions:
        #     print("手势预测结果:", gesture_predictions)

        #self.draw_finger_info(annotated_frame, detection_result,finger_counts, finger_statuses, gesture_predictions)
        gesture_direction = self.draw_gesture_info(annotated_frame, detection_result, frame.shape)
        self.draw_predictions(annotated_frame, number_predictions=number_predictions, gesture_predictions=gesture_predictions)
        cv2.imshow("Hand Tracking", annotated_frame)
        cv2.waitKey(1)

        return {
        "annotated_frame": annotated_frame,
        "gesture_direction": gesture_direction,
        "finger_counts": finger_counts,
        "finger_statuses": finger_statuses,
        "detection_result": detection_result,
        "number_predictions": number_predictions,
        "gesture_predictions": gesture_predictions,
        "hand_count": hand_count
    }
    def run(self, camera_id=0, window_name='Hand Tracking',
            draw_connections=True, draw_indices=False,
            show_finger_info=True, flip_frame=True):
        """运行主循环"""
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("按 'b' 键切换显示/隐藏连线")
        print("按 'n' 键切换显示/隐藏关键点编号")
        print("按 'f' 键切换显示/隐藏手指信息")
        print("按 'ESC' 键退出")

        show_connections = draw_connections
        show_indices = draw_indices
        show_info = show_finger_info

        # 用于显示历史轨迹
        trajectory_points = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break

            if flip_frame:
                frame = cv2.flip(frame, 1)
                #frame = cv2.flip(frame, 1)


            annotated_frame, detection_result, smoothed_hands = self.detect_hands(frame)
            finger_counts, finger_statuses = self.count_fingers(detection_result)
            positions = self.get_finger_positions(detection_result, frame.shape)

            self.draw_hand_landmarks(
                annotated_frame,
                detection_result,
                smoothed_hands,
                draw_connections=show_connections,
                draw_landmarks=True,
                draw_indices=show_indices
            )
            number_predictions = self.classify_hands(detection_result, smoothed_hands=smoothed_hands, predictor=self.predictor_number)
            gesture_predictions = self.classify_hands(detection_result, smoothed_hands=smoothed_hands, predictor=self.predictor_gesture)
            if show_info:
                #self.draw_finger_info(annotated_frame, detection_result,finger_counts, finger_statuses, number_predictions)
                self.draw_predictions(annotated_frame, number_predictions=number_predictions, gesture_predictions=gesture_predictions)

            # 检测并显示手势
            gesture_direction = self.draw_gesture_info(annotated_frame, detection_result, frame.shape)

            # 在手部位置绘制轨迹
            lcenter, rcenter = self.get_hand_center(detection_result, frame.shape)
            centers = [lcenter] if lcenter else [rcenter]
            for center in centers:
                trajectory_points.append(center)
                if len(trajectory_points) > 30:
                    trajectory_points.pop(0)

                # 绘制轨迹
                for i in range(1, len(trajectory_points)):
                    if trajectory_points[i - 1] and trajectory_points[i]:
                        cv2.line(annotated_frame, trajectory_points[i - 1],
                                 trajectory_points[i], (255, 255, 255), 2)

            if detection_result.hand_landmarks:
                cv2.putText(annotated_frame,
                            f"检测到 {len(detection_result.hand_landmarks)} 只手",
                            (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 显示方向指示（小提示）
            if gesture_direction != 'none' and gesture_direction != '检测中...':
                # 在屏幕边缘显示方向指示
                h, w = frame.shape[:2]
                color = (0, 255, 0) if gesture_direction in ['right', 'up'] else (0, 0, 255)
                direction_text = {
                    'left': '◀ 向左',
                    'right': '向右 ▶',
                    'up': '▲ 向上',
                    'down': '向下 ▼'
                }
                if gesture_direction in direction_text:
                    cv2.putText(annotated_frame, direction_text[gesture_direction],
                                (w - 150, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == ord('b'):
                show_connections = not show_connections
                print(f"连线显示: {'开启' if show_connections else '关闭'}")
            elif key == ord('n'):
                show_indices = not show_indices
                print(f"关键点编号显示: {'开启' if show_indices else '关闭'}")
            elif key == ord('f'):
                show_info = not show_info
                print(f"手指信息显示: {'开启' if show_info else '关闭'}")

        cap.release()
        cv2.destroyAllWindows()
    def get_training_data(self):
            """
            获取用于训练的数据

            Returns
            -------
            {
                "frame": 当前画面,
                "hand_detected": 是否检测到手,
                "hand_count": 手数量,
                "landmarks": [[x,y,z],...21个],
                "handedness": "Left"/"Right",
                "score": float
            }
            """

            result = self.process_frame()

            if result is None:
                return None

            detection_result = result["detection_result"]

            if len(detection_result.hand_landmarks) == 0:
                return {
                    "frame": result["annotated_frame"],
                    "hand_detected": False,
                    "hand_count": 0,
                    "landmarks": None,
                    "handedness": None,
                    "score": None
                }

            # 默认取第一只手
            hand_landmarks = detection_result.hand_landmarks[0]

            landmarks = []

            for lm in hand_landmarks:
                landmarks.append([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            handedness = None
            score = None

            if hasattr(detection_result, "handedness") and len(detection_result.handedness) > 0:
                handedness = detection_result.handedness[0][0].category_name
                score = detection_result.handedness[0][0].score

            return {
                "frame": result["annotated_frame"],
                "hand_detected": True,
                "hand_count": result["hand_count"],
                "landmarks": landmarks,
                "handedness": handedness,
                "score": score
            }

