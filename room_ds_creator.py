import os
import cv2
import numpy as np
import csv


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height
    if max_x == 0:
        x = 0
    else:
        x = np.random.randint(0, max_x)
    if max_y == 0:
        y = 0
    else:
        y = np.random.randint(0, max_y)
    crop = image[y: y + crop_height, x: x + crop_width]
    return crop


def get_two_crops(image, h, w, res):
    image1 = image[0:res, 0:res]
    image2 = image[h - res:h, w - res:w]
    return image1, image2


def resized_image(file_path, res):
    # resize short side to res, keep aspect ratio
    img = cv2.imread(file_path)
    if img is None:
        return None, None, None
    h, w, c = img.shape
    if h < w:
        # h is short -> new_h is res
        h_new = res
        w_new = int(w * h_new / h)
    else:
        # w is short -> new_w is res
        w_new = res
        h_new = int(h * w_new / w)
    dim = (w_new, h_new)
    img_new = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img_new, h_new, w_new


def write_image(output_dir, imname, im, write_flip):
    filename = os.path.join(output_dir, imname + ".png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(filename, im)

    # write horizontally flipped image
    if write_flip:
        filename = output_dir + imname + "_flip.png"
        cv2.imwrite(filename, cv2.flip(im, 1))


def transform_and_write(output_dir, f, res, ind, write_flip=False):
    # crop image (256 for train, 512 else)
    im, h, w = resized_image(f, res)
    if im is None:
        return ind

    ID = f[f.find("ID")+3:f.rfind(".")]

    # crop image and write
    r = w/h
    if r >= 4 / 3:
        im1, im2 = get_two_crops(im, h, w, res)
        write_image(output_dir, ID + "_1", im1, write_flip)
        write_image(output_dir, ID + "_2", im2, write_flip)
        ind = ind + 2
    else:
        im = get_random_crop(im, res, res)
        write_image(output_dir, ID, im, write_flip)
        ind = ind + 1
    return ind


root = r"C:\Users\PC\UR\www.houzz.com-photos"
new_root = r"C:\Users\PC\UR\room_classifier_houzz_dataset"

file = []
room = []
style = []
budget = []
name = []

for d in os.walk(root):
    if d[2]:
        for f in d[2]:
            ext = f[f.rfind('.')+1:]
            if ext in ['jpg', 'jpeg', 'png']:
                file.append(os.path.join(d[0], f))
                name.append(f[:f.rfind('.')])
                splits = d[0].split("\\")
                style.append(splits[-1])
                budget.append(splits[-2])
                room.append(splits[-3])

room_legend = list(set(room))
style_legend = list(set(style))
# budget_legend = list(set(budget))
budget_legend = ['Low Budget', 'Mid Low Budget',  'Mid High Budget', 'High Budget']

image_path = []
room_target = []
style_target = []
budget_target = []
room_type = []
style_type = []
budget_type = []
N_total = len(file)
N_corrupt = 0

for i in range(len(room)):
    im, h, w = resized_image(file[i], 512)
    if im is None:
        # image is corrupt
        N_corrupt += 1
        continue
    # im = get_random_crop(im, 512, 512) todo

    room_target.append(room_legend.index(room[i]))
    room_type.append(room[i])
    style_target.append(style_legend.index(style[i]))
    style_type.append(style[i])
    budget_target.append(budget_legend.index(budget[i]))
    budget_type.append(budget[i])

    filename = os.path.join(os.path.join(new_root, "images"), name[i] + ".png").replace(" ", "_")
    filepath = ("images/" + name[i] + ".png").replace(" ", "_")
    image_path.append(filepath)
    # cv2.imwrite(filename, im) todo
    # print("test")

# write csv
with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\data_fix.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "room", "room_target", "style", "style_target", "budget", "budget_target"])
    for i in range(len(image_path)):
        writer.writerow([image_path[i], room_type[i], room_target[i], style_type[i], style_target[i], budget_type[i], budget_target[i]])

# write legend
with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\legend.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(room_legend)
    writer.writerow(style_legend)
    writer.writerow(budget_legend)

print("done")
