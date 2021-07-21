import argparse
from datetime import datetime

import torch.optim as optim

from advance_model import AttU_Net
from base_model import UNet
from dataset import SEMDataset
from loss import bce_loss, dice_loss, dbce_loss, cross_entropy_loss
from modules import *
from save_history import *
from transform import random_transform_generator

PATH = None


def train(args):
    # transform generator
    if args.transform:
        print(args.transform)
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5,
        )
    else:
        print('no transformmmm')
        transform_generator = None

    # create custome dataset
    train_dataset = SEMDataset(os.path.join(args.train_dir, "imgs"),
                               os.path.join(args.train_dir, "labels"),
                               num_class=args.num_class,
                               target_h=480,
                               target_w=640,
                               transform_generator=transform_generator)
    val_dataset = SEMDataset(os.path.join(args.val_dir, "imgs"),
                             os.path.join(args.val_dir, "labels"),
                             target_h=480,
                             target_w=640,
                             num_class=args.num_class,
                             train=False)
    # Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, num_workers=args.num_workers,
                                               batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, num_workers=args.num_workers, batch_size=1,
                                             shuffle=False)
    # Model
    if args.model == "AttU_Net":
        model = AttU_Net(in_channels=3, n_classes=args.num_class)
    else:
        # default is classical UNet
        model = UNet(in_channels=3, n_classes=args.num_class, depth=3, batch_norm=True, padding=True)

    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    global PATH
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)
    # model = model.cuda()
    start_epoch = 1
    # if args.snapshot:
    if args.snapshot:
        start_epoch = load_checkpoint(args.snapshot, model, None, device=device)
    else:
        start_epoch = 1
    # model.load_state_dict(torch.load(PATH, map_location='cuda:0'))
    model = model.to(device)

    # Loss function
    if args.loss_fn == "bce":
        criterion = bce_loss
    elif args.loss_fn == "dice":
        criterion = dice_loss
    elif args.loss_fn == 'dbce':
        criterion = dbce_loss
    else:
        criterion = cross_entropy_loss

    # Saving History to csv
    header = ['epoch', 'train_loss', 'val_loss', 'val_acc']

    save_dir = os.path.join(args.save_dir, args.loss_fn)
    now = datetime.now()
    format_time = now.strftime("%Y_%m_%d_%H_%M")
    save_dir = os.path.join(save_dir, format_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    min_loss = 10000
    max_acc = 0
    min_loss_train = 10000
    try:
        for epoch in range(start_epoch, args.n_epoch):
            # train the model
            train_loss = train_model(epoch, model, train_loader, criterion, optimizer, scheduler, device)

            # validation every args.val_interval epoch

            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device, metric=True)
            print('Epoch %d, Train loss: %.5f, Val loss: %.5f, Val acc: %.4f' % (
                epoch, train_loss, val_loss, val_acc))

            values = [epoch, train_loss, val_loss, val_acc]
            export_history(header, values, save_dir, os.path.join(save_dir, "history.csv"))

            # save model every save_interval epoch
            if min_loss > val_loss or max_acc < val_acc or min_loss_train >= train_loss:
                save_checkpoint(os.path.join(save_dir, "{0}_{1}_{2:4f}.pth".format(args.model, epoch, train_loss)),
                                model, optimizer,
                                epoch)
                min_loss = min(val_loss, min_loss)
                max_acc = max(val_acc, max_acc)
                min_loss_train = min(min_loss_train, train_loss)

    except KeyboardInterrupt:
        save_checkpoint(os.path.join(save_dir, "{0}_final.pth".format(args.model)), model, optimizer, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--transform', help='', action='store_true')
    parser.add_argument('-l', '--loss_fn', type=str, default="cross_entropy_loss")
    parser.add_argument('-s', '--snapshot', type=str)
    parser.add_argument('-m', '--model', type=str, default="AttU_Net")
    parser.add_argument('-td', '--train_dir', type=str)
    parser.add_argument('-vd', '--val_dir', type=str)
    parser.add_argument('-sd', '--save_dir', type=str)
    parser.add_argument('-lr', '--lr', type=float, default=1e-4)
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-4)
    parser.add_argument('-ne', '--n_epoch', type=int, default=100)
    parser.add_argument('-bs', '--batch_size', type=int, default=8)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-vi', '--val_interval', type=int, default=1)
    parser.add_argument('-si', '--save_interval', type=int, default=3)
    parser.add_argument('-nc', '--num_class', type=int, default=5)
    parser.add_argument('-gid', '--gpu_id', type=int, default=0)

    args = parser.parse_args()
    train(args)
