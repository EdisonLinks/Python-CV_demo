import argparse
import functools
import os
import time
from collections import deque

import cv2
import numpy as np

import torch
from PIL import ImageDraw, ImageFont, Image

from detection.face_detect import MTCNN
from utils.utils import add_arguments, print_arguments
# 添加以下依赖
# pip install pyttsx3
# pip install websockets
import pyttsx3          # 监测人，人数统计人数，发送给web进行可视化
from websockets.sync.server import serve
import time
import datetime
import json
import random
import threading
import asyncio
import base64  # 添加base64导入

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('camera_id',                int,     0,                                  '使用的相机ID')
add_arg('face_db_path',             str,     'face_db',                          '人脸库路径')
add_arg('threshold',                float,   0.6,                                '判断相识度的阈值')
add_arg('mobilefacenet_model_path', str,     'save_model/mobilefacenet.pth',     'MobileFaceNet预测模型的路径')
add_arg('mtcnn_model_path',         str,     'save_model/mtcnn',                 'MTCNN预测模型的路径')
args = parser.parse_args()
print_arguments(args)

# 在 Predictor 类外添加全局变量
attendance_records = {
    'teachers': 0,
    'students': 0,
    'recognized_people': set(),  # 记录已识别的人员
    'last_recognition_time': {}  # 记录最后识别时间
}

class Predictor:
    def __init__(self, mtcnn_model_path, mobilefacenet_model_path, face_db_path, threshold=0.7):
        self.threshold = threshold
        self.mtcnn = MTCNN(model_path=mtcnn_model_path)
        # self.device = torch.device("cuda")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        self.model = torch.jit.load(mobilefacenet_model_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()

        self.faces_db = self.load_face_db(face_db_path)

    def load_face_db(self, face_db_path):
        faces_db = dict()
        # 添加一个身份映射配置
        role_mapping = {
            'teacher': ['老师', '教师', 'teacher', 'Teacher'],
            'student': ['学生', '同学', 'student', 'Student']
        }

        for path in os.listdir(face_db_path):
            name = os.path.basename(path).split('.')[0]
            # 添加判断：解析文件名格式：姓名_身份.jpg
            if '_' in name:
                real_name, role = name.split('_', 1)
                # 标准化身份标识
                role_lower = role.lower()
                if any(keyword in role_lower for keyword in role_mapping['teacher']):
                    role = 'teacher'
                else:
                    role = 'student'
            else:
                real_name = name
                role = 'student'

            image_path = os.path.join(face_db_path, path)
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
            imgs, _ = self.mtcnn.infer_image(img)
            imgs = self.process(imgs)
            feature = self.infer(imgs[0])
            faces_db[real_name] = {
                'feature': feature[0][0],
                'role': role
            }

        return faces_db

    @staticmethod
    def process(imgs):
        imgs1 = []
        for img in imgs:
            img = img.transpose((2, 0, 1))
            img = (img - 127.5) / 127.5
            imgs1.append(img)
        return imgs1

    # 预测图片
    def infer(self, imgs):
        assert len(imgs.shape) == 3 or len(imgs.shape) == 4
        if len(imgs.shape) == 3:
            imgs = imgs[np.newaxis, :]
        features = []
        for i in range(imgs.shape[0]):
            img = imgs[i][np.newaxis, :]
            img = torch.tensor(img, dtype=torch.float32, device=self.device)
            # 执行预测
            feature = self.model(img)
            feature = feature.detach().cpu().numpy()
            features.append(feature)
        return features

    def recognition(self, img):
        imgs, boxes = self.mtcnn.infer_image(img)
        if imgs is None:
            return None, None, 0

        # 统计至当前帧下的访客人数总和
        visitor_num = len(imgs)
        # 计算所有人脸的坐标对的距离
        distances = np.sqrt(np.square(boxes[:, 0] - boxes[:, 2]) + np.square(boxes[:, 1] - boxes[:, 3]))
        # 取最大值的索引，即取【最大人脸】的索引
        face_index = np.argsort(distances).tolist()[-1]
        # 提取这个【最大人脸】的人名，list的形式
        imgs = np.array([imgs[face_index]])
        boxes_list = boxes.tolist()
        box = boxes_list[face_index]
        boxes = np.array([box])

        imgs = self.process(imgs)
        imgs = np.array(imgs, dtype='float32')
        features = self.infer(imgs)

        names = []
        probs = []
        roles = []
        for i in range(len(features)):
            feature = features[i][0]
            results_dict = {}
            for name in self.faces_db.keys():
                # 修复：正确访问每个人脸的特征向量
                feature1 = self.faces_db[name]['feature']  # 正确访问方式
                prob = np.dot(feature, feature1) / (np.linalg.norm(feature) * np.linalg.norm(feature1))
                results_dict[name] = prob
            results = sorted(results_dict.items(), key=lambda d: d[1], reverse=True)
            print('人脸对比结果：', results)
            result = results[0]
            prob = float(result[1])
            probs.append(prob)
            if prob > self.threshold:
                name = result[0]
                names.append(name)
                roles.append(self.faces_db[name]['role'])  # 获取对应身份
            else:
                names.append('unknow')
                roles.append('unknow')
        return boxes, names, visitor_num, roles

    def add_text(self, img, text, left, top, color=(0, 0, 0), size=20):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype('simfang.ttf', size)
        draw.text((left, top), text, color, font=font)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # 画出人脸框和关键点
    def draw_face(self, img, boxes_c, names):
        if boxes_c is not None:
            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                name = names[i]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # 画人脸框
                cv2.rectangle(img, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                # 判别为人脸的名字
                img = self.add_text(img, name, corpbbox[0], corpbbox[1] -15, color=(0, 0, 255), size=12)
        return img  # 返回绘制后的图像，而不是显示它

# 文字转语音播放
# 使用 pyttsx3 本地语音引擎实现文字转语音。
# 修改为异步执行（可选优化），且采取全局初始化
engine = pyttsx3.init()
engine.setProperty('rate', 150)   # ·语速
engine.setProperty('volume', 0.9) # 音量
engine_lock = threading.Lock()  # 添加锁确保线程安全

def text_to_speech(sound_text):
    with engine_lock:
        try:
            engine.say(sound_text)
            engine.runAndWait()
        except RuntimeError as e:
            if "run loop already started" in str(e):
                pass  # 忽略已经运行的错误
            else:
                raise e
def text_to_speech_async(sound_text):
    threading.Thread(target=text_to_speech, args=(sound_text,), daemon=True).start()

# 本地摄像头调用检测
def predict(websocket):
    # 初始化预测器
    predictor = Predictor(args.mtcnn_model_path, args.mobilefacenet_model_path, args.face_db_path,
                          threshold=args.threshold)

    # 打开摄像头
    cap = cv2.VideoCapture(args.camera_id)

    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("摄像头已启动，按 'q' 键退出")

    # 初始化统计变量
    visitor_total = 0

    while True:
        ret, img = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        start = time.time()

        # 进行人脸识别
        recognition_result = predictor.recognition(img)
        if recognition_result[0] is not None and len(recognition_result) == 4:
            boxes, names, visitor_num, roles = recognition_result
        else:
            # 处理返回值数量不匹配或boxes为None的情况
            if recognition_result[0] is None:
                boxes, names, visitor_num = None, [], 0
            else:
                boxes, names, visitor_num = recognition_result[:3]
            roles = ['unknow'] * len(names) if names else []  # 默认身份为unknow

        visitor_total += visitor_num

        current_time = datetime.datetime.now()
        time_HMS = current_time.strftime("%H:%M:%S")

        # 如果有人脸被检测到，绘制边框和标签
        if boxes is not None and len(boxes) > 0 and names and len(names) > 0:
            img_with_boxes = predictor.draw_face(img.copy(), boxes, names)
        else:
            img_with_boxes = img

        # 组装统计数据
        # 修正后的数据组装代码
        data = {
            'time': time_HMS,
            'total_visitors': len(attendance_records['recognized_people']),  # 使用去重后的总人数
            'teachers': attendance_records['teachers'],
            'students': attendance_records['students'],
            'recognized_list': list(attendance_records['recognized_people'])
        }

        # 通过WebSocket发送数据
        try:
            # 编码当前图像为base64 - 使用绘制了边框的图像
            _, buffer = cv2.imencode('.jpg', img_with_boxes)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            # 将图像数据加入到发送的数据中
            data['image'] = img_base64

            websocket.send(json.dumps(data))
        except Exception as e:
            print(f"WebSocket发送消息失败: {e}")
            break

        print(
            f"时间: {time_HMS}, 访客总数: {visitor_total}, 教师: {attendance_records['teachers']}, 学生: {attendance_records['students']}")

        if boxes is not None and len(boxes) > 0 and names and len(names) > 0:
            face_name = names[0]
            face_role = roles[0] if len(roles) > 0 else 'unknow'

            if face_name != "unknow":
                # 更新考勤统计
                if face_name not in attendance_records['recognized_people']:
                    attendance_records['recognized_people'].add(face_name)
                    if face_role == 'teacher':
                        attendance_records['teachers'] += 1
                    else:
                        attendance_records['students'] += 1
                    attendance_records['last_recognition_time'][face_name] = current_time

                    print('预测的人脸位置：', boxes.astype('int32').tolist())
                    print('识别的人脸名称：', names)
                    print('总识别时间：%dms' % int((time.time() - start) * 1000))

                    # 生成欢迎语言并播报
                    if face_role == 'teacher':
                        sound_text = "%s老师，已签到" % face_name
                    else:
                        sound_text = "%s同学，已签到" % face_name
                    text_to_speech_async(sound_text)
                else:
                    # 已经识别过的人员
                    print('预测的人脸位置：', boxes.astype('int32').tolist())
                    print('识别的人脸名称：', names)
                    print('总识别时间：%dms' % int((time.time() - start) * 1000))
        else:
            print('未检测到人脸')

    # 释放资源
    cap.release()
    print("摄像头已关闭")

if __name__ == '__main__':
    # 开启WebSocket服务端
    with serve(predict, "localhost", 8765) as server:
        server.serve_forever()

# 需要连接WebSocket客户端才能触发图片处理