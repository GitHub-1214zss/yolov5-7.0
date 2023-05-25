# -*- coding: utf-8 -*-
import cv2
import os

'''
文件保存采用三级目录的方式，
path是文件的绝对路径，例如：从盘符开始的路径：D:/实验/test/zhaopan.jpg
dir是path路径下的某个文件夹的名称，
file是dir文件下某个视频文件的名称
注意：我们从文件夹里面复制下来的路径是D:/实验/test/zhaopan.jpg这样的，
但是代码中的路径应该是D:/实验/test/zhaopan.jpg，需要用反斜杠去写路径。
'''


def video_to_frames(path, dir, file, savepath):
    # VideoCapture视频读取类
    # 抽取帧数
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)

    # 将视频名称切分为名字和后缀MP4，放在一个列表里面
    file = file.split('.mp4')
    # 将列表里面的第一个元素取出来，就是不带后缀的名字
    file = file[0]

    n = 1
    # 30帧一秒，则此处为3秒切一次
    frametime = 30
    # 取出总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    for i in range(int(frames)):
        # ret是一个bool类型的数，当为True的时候将这帧照片取出保存在frame里面，反之不取出。
        ret, frame = videoCapture.read()

        if i % frametime == 0:
            # 截取的图片的绝对路径，这里我们要建立一个保存图片的文件夹，例如D:/截图，这里是将图片放在一个文件夹下
            filename = savepath + '/' + file + '_' + str(n) + '.jpg'
            # 如果不想将图片放在一个文件下，而是和源文件一样的目录结构用下面代码
            # filename = savepath + '/' + dir + '/' + file + '_' + str(n) + '.jpg'
            # folder = savepath + '/' + dir
            # if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建为文件夹
            #     os.makedirs(folder)

            # 将截取视频的图片保存到绝对路径下面
            cv2.imencode('.jpg', frame)[1].tofile(filename)
            print(filename)
            n += 1


"""
目录结构：
|Video
    |413
        |...mp3
        |...mp3
    |......
|......
"""
if __name__ == '__main__':
    # 保存文件夹的路径
    savepath = 'D:/Projects/data/steal/img'
    # 将一级目录下的所有文件夹的名称以列表的形式保存
    dirs = os.listdir("D:/Projects/data/steal/1")
    for dir in dirs:
        # 将二级目录和一级目录合并，取出二级目录下所有视频文件名称，保存到列表里面
        files = os.listdir("D:/Projects/data/steal/1/")
        # 得到每个视频文件的绝对路径，并利用切分函数对其进行切分，为了防止异常发生，在出现异常的时候可以继续运行该函数
        for file in files:
            path = 'D:/Projects/data/steal/1/{}'.format(dir, file)
            try:
                video_to_frames(path, dir, file, savepath)
            except Exception:
                continue
