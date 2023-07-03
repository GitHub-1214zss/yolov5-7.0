# 修改图片尺寸，修改图片名称
import os
from PIL import Image  # python3安装pillow库
import os.path
import glob

Source_address = r"D:\Projects\PythonProject\yolov5-6.0-steal\VOCdevkit\VOC2008\JPEGImages"
target_address = r"D:\Projects\PythonProject\yolov5-6.0-steal\VOCdevkit\VOC2008\JPEGImages"


class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        self.path = target_address  # 表示需要命名处理的文件夹
        self.save_path = target_address  # 保存重命名后的图片地址

    def rename(self, i):
        filelist = os.listdir(self.path)  # 获取文件路径
        total_num = len(filelist)  # 获取文件长度（个数）
        i = i  # 表示文件的命名是从i开始的
        for item in filelist:
            print(item)
            # 初始的图片的格式为jpg格式的（或者源文件是png格式及其他格式，后面的转换格式就可以调整为自己需要的格式即可）
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(
                    self.path), item)  # 当前文件中图片的地址
                dst = os.path.join(os.path.abspath(self.save_path), str(
                    i) + '.jpg')  # 处理后文件的地址和名称,可以自己按照自己的要求改进
                try:
                    os.rename(src, dst)
                    print('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue
        print('total %d to rename & converted %d jpgs' % (total_num, i))


def convertSize(jpgfile, outdir, width=64, height=128):  # 图片的大小256*256

    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)

        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


if __name__ == '__main__':
    # for jpgfile in glob.glob(Source_address+"/*.jpg"):  # 修改该文件夹下的jpg图片
    #     convertSize(jpgfile, target_address)  # 另存为的文件夹路径
    demo = BatchRename()
    demo.rename(1)  # 从i开始重新编号
