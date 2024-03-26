import cv2 as cv

# 打开原始视频文件
#original_video_path = "F:/科研/视频_2023.10.13/output_video_cut_3.avi"
#original_video_path = "F:/科研/视频_2023.7.24_60_20_25_60/C0006.MP4"  # 替换为原始视频文件路径
#original_video_path = "F:/科研/视频_2023.8.19_60_20_not_naive/output_video_cut.avi"
#original_video_path = "F:/科研/视频_2023.8.19_60_20_not_naive/C0019.MP4"
#original_video_path = "F:/科研/视频_2023.9.28/C0013.MP4"
#original_video_path = "F:/科研/视频_2023.10.13/C0014.MP4"
original_video_path = "F:/科研/视频_2024.03.03/C0027.MP4"
cap = cv.VideoCapture(original_video_path)

# 设置开始提取的时间点（以秒为单位）
start_time = 126 *60
end_time = 146*60

# 获取视频帧率和总帧数
frame_rate = int(cap.get(cv.CAP_PROP_FPS))
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# 设置开始和结束的帧数
start_frame = int(start_time * frame_rate)
end_frame = int(end_time * frame_rate)

# 创建输出视频的参数
#output_video_path = "F:/科研/视频_2023.9.28/output_video_cut_1.avi"
#output_video_path = "F:/科研/视频_2023.10.13/output_video_cut_3_cut.avi"  # 输出视频文件路径
output_video_path = "F:/科研/视频_2024.03.03/output_video_3.avi"
fourcc = cv.VideoWriter_fourcc(*'MJPG')
output_video = cv.VideoWriter(output_video_path, fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

# 循环读取帧并写入输出视频
cap.set(cv.CAP_PROP_POS_FRAMES, start_frame)
while cap.isOpened() and cap.get(cv.CAP_PROP_POS_FRAMES) <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    output_video.write(frame)

# 释放资源
cap.release()
output_video.release()
cv.destroyAllWindows()
