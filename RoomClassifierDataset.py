import torch
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import csv
import os
import torch.nn.functional as F


class RoomClassifierDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        self.transform = transform
        self.csv = data_csv

        self.images = []
        self.room_labels = []
        self.room_targets = []
        self.budget_labels = []
        self.budget_targets = []
        self.style_labels = []
        self.style_targets = []

        with open(data_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for ind, row in enumerate(csv_reader):
                if ind == 0:
                    continue
                else:
                    self.images.append(os.path.join(os.path.dirname(data_csv), row[0].replace('/', '\\')))
                    self.room_labels.append(row[1])
                    self.room_targets.append(F.one_hot(torch.tensor(int(row[2])), 7).type(torch.float32))
                    self.style_labels.append(row[3])
                    self.style_targets.append(F.one_hot(torch.tensor(int(row[4])), 19).type(torch.float32))
                    self.budget_labels.append(row[5])
                    self.budget_targets.append(F.one_hot(torch.tensor(int(row[6])), 4).type(torch.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        room_target = self.room_targets[idx]
        style_target = self.style_targets[idx]
        budget_target = self.budget_targets[idx]

        return img, room_target, style_target, budget_target


class EmptyRoomClassifierDataset(Dataset):
    def __init__(self, data_csv, transform=None):
        self.transform = transform
        self.csv = data_csv

        self.images = []
        self.room_targets = []
        self.budget_targets = []
        self.style_targets = []
        self.empty = []

        with open(data_csv) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for ind, row in enumerate(csv_reader):
                if ind == 0 or ind < 350000: # TODO: remove <350000
                # if ind == 0:
                    continue
                else:
                    self.images.append(os.path.join(os.path.dirname(data_csv), row[0].replace('/', '\\')))
                    self.style_targets.append(F.one_hot(torch.tensor(int(row[4])), 19).type(torch.float32))
                    self.budget_targets.append(F.one_hot(torch.tensor(int(row[6])), 4).type(torch.float32))

                    if row[1] == "Empty":
                        self.room_targets.append(F.one_hot(torch.tensor(0), 7).type(torch.float32))
                        self.empty.append(torch.tensor(1).type(torch.float32))
                    else:
                        self.room_targets.append(F.one_hot(torch.tensor(int(row[2])), 7).type(torch.float32))
                        self.empty.append(torch.tensor(0).type(torch.float32))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        room_target = self.room_targets[idx]
        style_target = self.style_targets[idx]
        budget_target = self.budget_targets[idx]
        empty = self.empty[idx]

        return img, room_target, style_target, budget_target, empty


