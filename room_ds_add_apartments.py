import os
import cv2
import numpy as np
import csv
from tqdm import tqdm


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
    img_new = get_random_crop(img_new, 512, 512)
    return img_new, h_new, w_new


def write_image(output_dir, imname, im, write_flip=False):
    filename = os.path.join(output_dir, imname + ".png")
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(filename, im)

    # write horizontally flipped image
    if write_flip:
        filename = output_dir + imname + "_flip.png"
        cv2.imwrite(filename, cv2.flip(im, 1))


root = r"C:\Users\PC\UR\UR_lama_apartments_220502_only_ROOM_Tal_partial_clean\train"
image_folder = r"C:\Users\PC\UR\room_classifier_houzz_dataset\images"

image_paths = []
room_type = []
room_target = []
style_type = []
style_target = []
budget_type = []
budget_target = []
budget_legend = ['Low_Budget', 'Mid_Low_Budget',  'Mid_High_Budget', 'High_Budget']

# find all files in root
files = []
names = []
for d in os.walk(root):
    if d[2]:
        for f in d[2]:
            ext = f[f.rfind('.')+1:]
            if ext in ['jpg', 'jpeg', 'png']:
                files.append(os.path.join(d[0], f))
                names.append(f[:f.rfind('.')])

# read csv
with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\data_with_empty.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for ind, row in enumerate(csv_reader):
        if ind == 0:
            row0 = row
        else:
            image_paths.append(row[0])
            room_type.append(row[1])
            room_target.append(row[2])
            style_type.append(row[3])
            style_target.append(row[4])
            budget_type.append(row[5])
            budget_target.append(row[6])
    print(f'Processed {ind} lines.')

# for each file in root
for ind in tqdm(range(len(files))):
    # add resized image (512) to image folder
    im, h, w = resized_image(files[ind], 512)
    if im is not None:
        write_image(image_folder, names[ind], im)
        # add to lists
        image_paths.append("images/" + names[ind] + '.png')
        room_type.append("Empty")
        room_target.append(7)
        style_type.append("None")
        style_target.append(0)
        budget_type.append("None")
        budget_target.append(0)

# write csv
with open(r"C:\Users\PC\UR\room_classifier_houzz_dataset\data_with_empty_with_apartments.csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "room", "room_target", "style", "style_target", "budget", "budget_target"])
    for i in range(len(image_paths)):
        writer.writerow([image_paths[i], room_type[i], room_target[i], style_type[i], style_target[i], budget_type[i], budget_target[i]])