调用测试图片数据库进行人脸识别 

`python infer.py --image_path=/dataset/test.jpg`



调用摄像头进行人脸识别

python infer_camera.py --camera_id=0



文件树：

dataset  人脸识别测试数据库

detection MTCNN模型训练文件

face_db  人脸数据库

models  mobilefacenet模型训练文件

save_model  保存模型文件



图片人脸识别   infer.py

摄像头人脸识别 infer_camera.py