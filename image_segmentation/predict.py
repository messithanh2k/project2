import argparse

from tqdm import tqdm

from advance_model import AttU_Net
from base_model import UNet
from dataset import SEMValDataset
from modules import *
from save_history import *


def predict(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = SEMValDataset(os.path.join(args.input_dir))
    loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=args.num_workers, batch_size=1, shuffle=False)

    device = torch.device("cuda:%d" % args.gpu_id if torch.cuda.is_available() else "cpu")
    # init model
    if args.model == "AttU_Net":
        model = AttU_Net(in_channels=3, n_classes=args.num_class)
    else:
        # default is classical UNet
        model = UNet(in_channels=3, n_classes=args.num_class, depth=2, batch_norm=True, padding=True)
    model = model.to(device)

    load_checkpoint(args.snapshot, model, None)

    model.eval()
    with torch.no_grad():
        pbar = tqdm(loader)
        for batch_idx, images in enumerate(pbar):
            images = images.to(device)

            probs = model.forward(images).data.cpu().numpy()  # 1 * C * H * W
            preds = np.argmax(probs, axis=1).astype(np.uint8)  # 1 * H * W
            pred = preds[0, ...]  # H x W

            label = Image.fromarray(pred).convert("L")

            basename = dataset.get_basename(batch_idx)
            np.save(os.path.join(args.save_dir, "%s.npy" % basename), pred)
            # label.save(os.path.join(args.save_dir, "%s.png" % basename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="AttU_Net")
    parser.add_argument('-id', '--input_dir', type=str)
    parser.add_argument('-sd', '--save_dir', type=str)
    parser.add_argument('-s', '--snapshot', type=str)
    parser.add_argument('-nw', '--num_workers', type=int, default=0)
    parser.add_argument('-nc', '--num_class', type=int, default=5)
    parser.add_argument('-gid', '--gpu_id', type=int, default=0)
    args = parser.parse_args()

    predict(args)
