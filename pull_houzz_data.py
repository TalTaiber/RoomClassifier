import os

root = "/media/wonder/Transcend/crawlerDLs/www.houzz.com-photos"
subfolders = [f.path for f in os.scandir(root) if f.is_dir()]
rooms = [f[f.rfind("/")+1:] for f in subfolders]

dest = "/home/wonder/UR/houzz"
N = 10000

# take N images from each room
# for each room, take N/4 images from each budget
# for each room+budget, take images s.t. keep the proportion of styles

print("ok")