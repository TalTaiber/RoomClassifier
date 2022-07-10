import torch
from RoomClassifierDataset import RoomClassifierDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import csv
from RoomClassifierModule import RoomClassifierMobileNet
from RoomClassifierModule import evaluate, fit_one_cycle
import json


# helper functions to load the data and model onto GPU
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


device = get_default_device()
torch.manual_seed(0)  # reproducible

data_csv = r'C:\Users\PC\UR\room_classifier_houzz_dataset\data_fix.csv'
legend_csv = r'C:\Users\PC\UR\room_classifier_houzz_dataset\legend.csv'
with open(legend_csv) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for ind, row in enumerate(csv_reader):
        if ind == 0:
            room_legend = row
        elif ind == 1:
            style_legend = row
        elif ind == 2:
            budget_legend = row

imagenet_stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # mean and std values of the Imagenet Dataset so that
# pretrained models could also be used

# setting a set of transformations to transform the images
# transform = T.Compose([T.Resize(128),
#                        T.RandomCrop(128),
#                        T.RandomHorizontalFlip(),
#                        T.RandomRotation(2),
#                        T.ToTensor(),
#                        T.Normalize(*imagenet_stats)])
transform = T.Compose([T.Resize(256),
                       T.RandomHorizontalFlip(),
                       T.ToTensor(),
                       T.Normalize(*imagenet_stats)])

# create dataset
dataset = RoomClassifierDataset(data_csv, transform)
print(len(dataset))


# this function will denormalize the tensors
def denorm(img_tensors):
    return img_tensors * imagenet_stats[1][0] + imagenet_stats[0][0]


def show_example(img, room_target, style_target, budget_target):
    plt.imshow(denorm(img).permute(1, 2, 0))
    plt.title("Room: " + room_legend[room_target.argmax()] + "\nStyle: " + style_legend[style_target.argmax()] + ", Budget: " + budget_legend[budget_target.argmax()])
    plt.show()


show_example(*dataset[0])  # let's take an example

# val_percent = int(0.15 * len(dataset))  # setting 15 percent of the total number of images for validation
val_percent = int(0.15 * len(dataset))
train_size = len(dataset) - val_percent
val_size = len(dataset) - train_size

# todo: implement deterministically
train_ds, val_ds = random_split(dataset, [train_size, val_size])  # splitting the dataset for training and validation

# setting batch size for Dataloader to load the data batch by batch
batch_size = 32
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size * 2)

# loading training and validation data onto GPU
train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)

# create net (based on mobilenetv3)
model = RoomClassifierMobileNet(freeze_backbone=False).to(device)
model.load_state_dict(torch.load(r'C:\Users\PC\PycharmProjects\RoomClassifier\models\room_classifier.pth'))
for param in model.parameters():
    param.requires_grad = True

# test net output shapes
for x, r, s, b in train_dl:
    print(x.shape)
    print(r.shape)
    print(s.shape)
    print(b.shape)
    r_pred, s_pred, b_pred = model(x)
    print(r_pred.shape)
    print(s_pred.shape)
    print(b_pred.shape)
    break

# verify model working ok
history = []
print("Running validation with initial network...")
# history += [evaluate(model, val_dl)]

# define training hyperparams
epochs = 30
max_lr = 1e-3 # default 1e-3
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam

# run training
print("Training...")
history += fit_one_cycle(epochs, max_lr, model, train_dl, val_dl,
                         grad_clip=grad_clip,
                         weight_decay=weight_decay,
                         opt_func=opt_func)

# save model and history after training
torch.save(model.state_dict(), r"C:\Users\PC\PycharmProjects\RoomClassifier\models\room_classifier_60iter_nosoftmax.pth")
with open('history.json', 'w') as f:
    json.dump(history, f)


def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()


# plot losses
plot_losses(history)

print("Training done.")
