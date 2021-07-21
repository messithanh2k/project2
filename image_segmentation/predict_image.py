import cv2
from torchvision import transforms

from advance_model import AttU_Net
from modules import *
from pre_processing import normalize_image
from save_history import *

SIZE = (640, 480)


def val_resize_crop_padding(image):
    target_dim = SIZE  # width x height
    scale_factor = float(target_dim[0]) / image.shape[1]
    # resize to fit target width
    image = cv2.resize(image, (target_dim[0], int(image.shape[0] * scale_factor)), interpolation=cv2.INTER_NEAREST)
    if image.shape[0] < target_dim[1]:
        pad_size = abs(target_dim[1] - image.shape[0]) // 2
        image = cv2.copyMakeBorder(image, pad_size, pad_size, 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
    image = cv2.resize(image, target_dim, interpolation=cv2.INTER_NEAREST)
    return image


snapshot = "model/project/AttU_Net_30_3.392057.pth"
gpu_id = 0
num_class = 5


def predict(image):
    transform = transforms.ToTensor()
    device = torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")
    # init model
    model = AttU_Net(in_channels=3, n_classes=num_class)
    model = model.to(device)

    load_checkpoint(snapshot, model, None)

    model.eval()

    with torch.no_grad():
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = val_resize_crop_padding(image)

        image = normalize_image(image)
        image = transform(image)
        image = torch.reshape(image, (1, 3, 480, 640))
        image = image.to(device)

        probs = model.forward(image).data.cpu().numpy()  # 1 * C * H * W
        preds = np.argmax(probs, axis=1).astype(np.uint8) # 1 * H * W
        pred = preds[0, ...]  # H x W

    return pred


# image = cv2.imread("real_image/IMG_20210528_095253.jpg")
# inp = predict(image)
# gray = np.where(inp == 0, inp, 255)
# print(np.unique(inp, return_counts=True))
# cv2.imshow("test", gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()