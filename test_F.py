import torch
from RoomClassifierDataset import EmptyRoomClassifierDataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
import csv
from RoomClassifierModule import EmptyRoomClassifierMobileNet
from RoomClassifierModule import evaluate, fit_one_cycle
import json
import easygui as eg
import torch.nn as nn
from PIL import Image


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

data_csv = r'C:\Users\PC\UR\room_classifier_houzz_dataset\data_with_empty.csv'
legend_csv = r'C:\Users\PC\UR\room_classifier_houzz_dataset\legend.csv'
load_path = r'C:\Users\PC\PycharmProjects\RoomClassifier\models\room_classifier_90iter_all_grads_empty_full_ds.pth'

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
dataset = EmptyRoomClassifierDataset(data_csv, transform)


# this function will denormalize the tensors
def denorm(img_tensors):
    return img_tensors * imagenet_stats[1][0] + imagenet_stats[0][0]


def show_example(img, room_target, style_target, budget_target):
    plt.imshow(denorm(img).permute(1, 2, 0))
    plt.title("Room: " + room_legend[room_target.argmax()] + "\nStyle: " + style_legend[style_target.argmax()] + ", Budget: " + budget_legend[budget_target.argmax()])
    plt.show()


def show_test_example(img, r_t, r_p, s_t, s_p, b_t, b_p, e_t, e_p):
    # todo: add loss
    # loss_r = nn.CrossEntropyLoss()(r_pred, r_target)
    # loss_s = nn.CrossEntropyLoss()(s_pred, s_target)
    # loss_b = nn.CrossEntropyLoss()(b_pred, b_target)
    #loss_r = nn.BCEWithLogitsLoss()(r_pred, r_target)
    #loss_s = nn.BCEWithLogitsLoss()(s_pred, s_target)
    #loss_b = nn.BCEWithLogitsLoss()(b_pred, b_target)

    loss_r = nn.BCEWithLogitsLoss(reduction='none')(r_p, r_t)
    loss_r = torch.dot(loss_r.mean(1), 1 - e_t) / (len(e_t) - sum(e_t))
    loss_s = nn.BCEWithLogitsLoss(reduction='none')(s_p, s_t)
    loss_s = torch.dot(loss_s.mean(1), 1 - e_t) / (len(e_t) - sum(e_t))
    loss_b = nn.BCEWithLogitsLoss(reduction='none')(b_p, b_t)
    loss_b = torch.dot(loss_b.mean(1), 1 - e_t) / (len(e_t) - sum(e_t))
    loss_e = nn.BCEWithLogitsLoss()(e_p, e_t)

    if r_p.sigmoid().argmax() == r_t.argmax():
        print("ok")

    F = F_score(r_p, r_t)

    plt.imshow(denorm(img).permute(1, 2, 0))
    string = str(
        room_legend[r_t.argmax()] + " /  " + room_legend[r_p.sigmoid().argmax()] + f" ({(r_p.sigmoid().max().cpu().numpy() * 100):2.0f}%)" + f" loss = {loss_r.cpu().numpy():2.2f}"
        + "\n" + style_legend[s_t.argmax()] + " / " + style_legend[s_p.sigmoid().argmax()] + f" ({(s_p.sigmoid().max().cpu().numpy() * 100):2.0f}%)" + f" loss = {loss_s.cpu().numpy():2.2f}"
        + "\n" + budget_legend[b_t.argmax()] + " / " + budget_legend[b_p.sigmoid().argmax()] + f" ({(b_p.sigmoid().max().cpu().numpy() * 100):2.0f}%)" + f" loss = {loss_b.cpu().numpy():2.2f}"
        + "\n" + "Empty: " + str(bool(e_t)) + " / " + str(e_p.sigmoid().cpu().numpy()[0] > 0.5) + f" ({(e_p.sigmoid().max().cpu().numpy() * 100):2.0f}%)" + f" loss = {loss_e.cpu().numpy():2.2f}"
        + "\n" + f"F_score = {F:1.3f}"
    )

    # # show predictions
    # print("Room predictions:")
    # for ind in range(len(r_p[0])):
    #     print(f"\t{room_legend[ind]} ({100*r_p.sigmoid()[0][ind]:2.2f}%)")
    # print("Style predictions:")
    # for ind in range(len(s_p[0])):
    #     print(f"\t{style_legend[ind]} ({100 * s_p.sigmoid()[0][ind]:2.2f}%)")
    # print("Budget predictions:")
    # for ind in range(len(b_p[0])):
    #     print(f"\t{budget_legend[ind]} ({100 * b_p.sigmoid()[0][ind]:2.2f}%)")

    plt.text(20, 10, string, bbox=dict(fill=True, edgecolor='yellow', linewidth=2, facecolor='white'), size=10)
    plt.show()


def F_score(pred, target):
    TP = torch.sum(pred.sigmoid()*target, dim=0)
    FN = torch.sum((1-pred.sigmoid())*target, dim=0)
    FP = torch.sum(pred.sigmoid()*(1-target), dim=0)
    TN = torch.sum((1-pred.sigmoid())*(1-target), dim=0)

    # acc_smooth = (TP+TN)/(FP+FN+TP+TN)
    prec_smooth = (TP)/(FP+TP)
    rec_smooth = (TP)/(FN+TP)

    F_smooth = 2*(prec_smooth*rec_smooth)/(prec_smooth+rec_smooth)

    return F_smooth.mean()


# show_example(*dataset[0])  # let's take an example

# setting batch size for Dataloader to load the data batch by batch
batch_size = 1
test_loader = DataLoader(dataset, batch_size, shuffle=True)

# loading test data onto GPU
test_dl = DeviceDataLoader(test_loader, device)

# create net (based on mobilenetv3)
model = EmptyRoomClassifierMobileNet(freeze_backbone=False).to(device)
model.load_state_dict(torch.load(load_path))
for param in model.parameters():
    param.requires_grad = False

# change net to test mode
model.eval()

# test net output
for x, r, s, b, e in test_dl:
    r_pred, s_pred, b_pred, e_pred = model(x)
    show_test_example(x.squeeze().cpu(), r, r_pred, s, s_pred, b, b_pred, e, e_pred)  # let's take an example
    if eg.ccbox(msg="Test another image?"):  # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:  # user chose Cancel
        break


# test image from pc
def test_image_from_pc():
    im_path = eg.fileopenbox(default=r"C:\Users\PC\Desktop\empty rooms\\")
    img = Image.open(im_path)
    img = transform(img).to(device).unsqueeze(0)
    r_pred, s_pred, b_pred, e_pred = model(img)
    show_test_example(img.squeeze().cpu(), r, r_pred, s, s_pred, b, b_pred, e, e_pred)

while(True):
    test_image_from_pc()
    if eg.ccbox(msg="Test another image?"):  # show a Continue/Cancel dialog
        pass  # user chose Continue
    else:  # user chose Cancel
        break

# verify model on entire test set
# print("Running validation with initial network...")
# history = [evaluate(model, test_dl)]

# def plot_losses(history):
#     train_losses = [x.get('train_loss') for x in history]
#     val_losses = [x['val_loss'] for x in history]
#     plt.plot(train_losses, '-bx')
#     plt.plot(val_losses, '-rx')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Training', 'Validation'])
#     plt.title('Loss vs. No. of epochs')
#     plt.show()


# plot losses
# plot_losses(history)


print("test done")
