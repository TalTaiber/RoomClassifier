import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import torchvision

# Calculate the accuracy of the model
def F_score(prediction, target):
    # todo: write multiclass F score
    return torch.Tensor(0)


class RoomClassifierBase(nn.Module):
    def training_step(self, batch):
        images, r_t, s_t, b_t, e_t = batch
        r_p, s_p, b_p, e_p = self(images)  # Generate predictions

        # loss_r = nn.BCEWithLogitsLoss()(r_p, r_t)
        # loss_s = nn.BCEWithLogitsLoss()(s_p, s_t)
        # loss_b = nn.BCEWithLogitsLoss()(b_p, b_t)
        # loss_e = nn.BCEWithLogitsLoss()(e_p.flatten(), e_t)

        loss_r = nn.BCEWithLogitsLoss(reduction='none')(r_p, r_t)
        loss_r = torch.dot(loss_r.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_s = nn.BCEWithLogitsLoss(reduction='none')(s_p, s_t)
        loss_s = torch.dot(loss_s.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_b = nn.BCEWithLogitsLoss(reduction='none')(b_p, b_t)
        loss_b = torch.dot(loss_b.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_e = nn.BCEWithLogitsLoss()(e_p, e_t)

        loss = loss_r + loss_s + loss_b + loss_e

        return loss

    def validation_step(self, batch):
        images, r_t, s_t, b_t, e_t = batch
        r_p, s_p, b_p, e_p = self(images)  # Generate predictions

        # loss_r = nn.BCEWithLogitsLoss()(r_p, r_t)
        # loss_s = nn.BCEWithLogitsLoss()(s_p, s_t)
        # loss_b = nn.BCEWithLogitsLoss()(b_p, b_t)
        # loss_e = nn.BCEWithLogitsLoss()(e_p.flatten(), e_t)

        loss_r = nn.BCEWithLogitsLoss(reduction='none')(r_p, r_t)
        loss_r = torch.dot(loss_r.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_s = nn.BCEWithLogitsLoss(reduction='none')(s_p, s_t)
        loss_s = torch.dot(loss_s.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_b = nn.BCEWithLogitsLoss(reduction='none')(b_p, b_t)
        loss_b = torch.dot(loss_b.mean(1), 1-e_t)/(len(e_t) - sum(e_t))
        loss_e = nn.BCEWithLogitsLoss()(e_p, e_t)

        loss = loss_r + loss_s + loss_b + loss_e

        # todo: finish
        score_r = F_score(1, 1)

        return {'val_loss': loss.detach(), 'val_room_score': score_r.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()       # Combine losses and get the mean value
        batch_scores = [x['val_room_score'] for x in outputs]
        epoch_score = torch.stack(batch_scores).mean()      # Combine accuracies and get the mean value
        return {'val_loss': epoch_loss.item(), 'val_score': epoch_score.item()}

    # display the losses
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.4f}, train_loss: {:.4f}, val_loss: {:.4f}".format(epoch, result['lrs'][-1], result['train_loss'], result['val_loss']))


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, save_path=None, weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)) #schedule the learning rate with OneCycleLR

    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        lrs = []
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()

        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)

        # save model
        if save_path:
            torch.save(model.state_dict(), save_path)
    return history


class RoomClassifierMobileNet(RoomClassifierBase):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v3_large(weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)

        # set all params to no grad
        if freeze_backbone:
            for param in mobilenet.parameters():
                param.requires_grad = False

        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool

        # create 3 new classifiers (with similar arch to mobilenet classifier)
        self.room_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 7, bias=True),  # TODO: made output dim not hard-coded
            # nn.Softmax(1)
        )
        self.style_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 19, bias=True),  # TODO: made output dim not hard-coded
            # nn.Softmax(1)
        )
        self.budget_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 4, bias=True),  # TODO: made output dim not hard-coded
            # nn.Softmax(1)
        )

    def forward(self, x):
        # extract features from mobilenet
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)

        # pass features to classifiers
        r = self.room_classifier(f)
        s = self.style_classifier(f)
        b = self.budget_classifier(f)

        return r, s, b


class EmptyRoomClassifierMobileNet(RoomClassifierBase):
    def __init__(self, freeze_backbone=False):
        super().__init__()
        mobilenet = torchvision.models.mobilenet_v3_large(
            weights=torchvision.models.MobileNet_V3_Large_Weights.DEFAULT)

        # set all params to no grad
        if freeze_backbone:
            for param in mobilenet.parameters():
                param.requires_grad = False

        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool

        # create 3 new classifiers (with similar arch to mobilenet classifier)
        self.room_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 7, bias=True)  # TODO: made output dim not hard-coded
        )
        self.style_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 19, bias=True)  # TODO: made output dim not hard-coded
        )
        self.budget_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 4, bias=True)  # TODO: made output dim not hard-coded
        )
        self.empty_classifier = nn.Sequential(
            nn.Linear(960, 1280, bias=True),
            nn.Hardswish(),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(1280, 1, bias=True)  # TODO: made output dim not hard-coded
        )

    def forward(self, x):
        # extract features from mobilenet
        f = self.features(x)
        f = self.avgpool(f)
        f = torch.flatten(f, 1)

        # pass features to classifiers
        r = self.room_classifier(f)
        s = self.style_classifier(f)
        b = self.budget_classifier(f)
        e = self.empty_classifier(f).flatten()

        return r, s, b, e
