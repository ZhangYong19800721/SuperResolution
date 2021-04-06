import os

def file_name(in_video_dir, out_image_dir):
    for root, dirs, files in os.walk(in_video_dir):
        print('root_dir:', root)  # 当前目录路径
        print('sub_dirs:', dirs)  # 当前路径下所有子目录
        print('files:', files)  # 当前路径下所有非目录子文件
        for file in files:
            if file.find(".mp4") != -1:
                file_name_without_ext = file.replace(".mp4","")
                command = f"/home/zhangyong/Program/ffmpeg-4.3.1-amd64-static/ffmpeg -i {root}/{file} -f image2 -vf fps=fps=1/3 -qscale:v 2 {out_image_dir}/{file_name_without_ext}-image-%06d.jpeg"
                os.system(command)

if __name__ == "__main__":
    file_name("/home/zhangyong/Downloads/video4K/", "/home/zhangyong/Data/image2160x3840")