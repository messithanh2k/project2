import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from augment_data import *
from pre_processing import *


def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def add_noise(img):
    VARIABILITY = 50
    deviation = VARIABILITY * random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img


def show_img(name, mat):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, mat)
    cv2.waitKey()


class SEMDataset(Dataset):
    def __init__(self,
                 image_dir,
                 label_dir,
                 temp_dir="tmp",
                 num_class=2,
                 target_h=None,
                 target_w=None,
                 transform_generator=None,
                 transform_parameters=None,
                 train=True):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.temp_dir = temp_dir

        self.num_class = num_class
        self.transform = transforms.ToTensor()

        self.transform_generator = transform_generator
        self.transform_parameters = transform_parameters or TransformParameters()

        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_paths.sort()

        self.label_paths = [os.path.join(label_dir, "%s.png" % os.path.basename(filepath).split(".")[0]) for filepath in
                            self.image_paths]

        self.target_w = target_w
        self.target_h = target_h
        self.train=train

    def get_basename(self, index):
        basename = os.path.basename(self.image_paths[index]).split(".")[0]
        return basename

    def resize_images(self, images, size):
        for idx, img in enumerate(images):
            img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
            images[idx] = img
        return images

    def _augment(self, image, label, conf,ratio_fg=1):
        rnd=random.random()
        if rnd < 0:
            jitter = self.color_jitter_image()
            # print("image shape: ",image.shape)
            image = jitter(transforms.ToPILImage()(image))
            image = np.array(image)
        elif rnd < 0.6:
            image = self.augment_data(image)
        random_transforms = [image, label, conf]
        rnd=random.random()
        if ratio_fg<0.7 and rnd<0.3:
            random_transforms=self.random_scale(random_transforms)
        img_h, img_w = random_transforms[0].shape[:2]
        rnd=random.random()
        if rnd<0.3 and ratio_fg<1/2 and img_h>self.target_h and img_w>self.target_w:
            random_transforms = self.random_crop(random_transforms, (self.target_h, self.target_w))
        else:
            random_transforms = self.resize_images(random_transforms, (self.target_w, self.target_h))
        random_transforms = self.random_horizontal_flip(random_transforms)
        random_transforms = self.random_rotate(random_transforms)
        image, label, conf = random_transforms
        return image, label

    def __getitem__(self, index):
        # pre-processing image to target dim
        # print("index: ",index )
        if self.train:
            image, mask, conf = load_data(self.image_paths[index], self.label_paths[index],train=True)
            # online augmentation
            nb_pixel = image.shape[0] * image.shape[1]
            nb_foreground = np.sum(mask)
            ratio_foreground = nb_foreground / nb_pixel

            image, mask = self._augment(image, mask, conf,ratio_foreground)
        else:
            image, mask,conf = load_data(self.image_paths[index], self.label_paths[index], train=False)
            image=self.resize_images([image], (self.target_w, self.target_h))[0]
            mask=self.resize_images([mask], (self.target_w, self.target_h))[0]
            conf=self.resize_images([conf], (self.target_w, self.target_h))[0]
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # print("shapeeeee: ",gray_img.shape,mask.shape)
        out = np.hstack((gray_img, (mask * 255).astype(np.uint8)))
        img_path = self.image_paths[index]
        img_fname = os.path.basename(img_path)
        cv2.imwrite("debug/" + img_fname, out)
        image = normalize_image(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        label = torch.from_numpy(mask).long()
        return image, label

    def __len__(self):
        return len(self.image_paths)
        # return 1

    def color_jitter_image(self):
        color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.4)
        transform = transforms.ColorJitter.get_params(
            color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
            color_jitter.hue)
        return transform

    def random_scale(self, imgs, size=(640, 480), min_scale=0.3, rnd_scale=1):
        if random.random() < rnd_scale:
            h, w = imgs[0].shape[0:2]
            size_w, size_h = size[0] * 1.2, size[1] * 1.2
            if h > size_h or w > size_w:
                max_scale = max(size_h / h, size_w / w)
                scale = random.uniform(min_scale, max_scale)
            else:
                max_scale = max(h / size_h, w / size_w)
                scale = random.uniform(min_scale * 2, max_scale)
            # print("scale: ", scale, h, w)
            for idx, img in enumerate(imgs):
                img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
                # print("shape scale: ",img.shape)
                imgs[idx] = img
        return imgs

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size, self.target_size))

    def padding_image(self, images, target_h, target_w):
        padded_imgs = []
        actual_h, actual_w = images[0].shape[:2]
        target_h = max(target_h, actual_h)
        target_w = max(target_w, actual_w)
        pad_h = target_h - actual_h
        pad_w = target_w - actual_w
        rnd_pad_top, rnd_pad_bot, rnd_pad_right, rnd_pad_left = 0, 0, 0, 0
        if pad_h > 0:
            rnd_pad_top = random.randint(0, pad_h)
            rnd_pad_bot = pad_h - rnd_pad_top
        if pad_w > 0:
            rnd_pad_left = random.randint(0, pad_w)
            rnd_pad_right = pad_w - rnd_pad_left
        for idx, image in enumerate(images):
            mean = int(np.mean(image))
            input_dimension = len(image.shape)
            target_shape = (target_h, target_w) if input_dimension == 2 else (target_h, target_w, 3)
            big_img = np.ones(target_shape, dtype=np.uint8) * mean if idx == 0 else np.zeros(target_shape,
                                                                                             dtype=np.uint8)
            big_img[rnd_pad_top:target_h - rnd_pad_bot, rnd_pad_right:target_w - rnd_pad_left] = image
            padded_imgs.append(big_img)
        return padded_imgs,rnd_pad_top,rnd_pad_left

    def random_crop(self, imgs, img_size, scale=1.2):
        target_h, target_w = img_size
        padded_h, padded_w = int(target_h * scale), int(target_w * scale)
        imgs,pad_top,pad_left = self.padding_image(imgs, padded_h, padded_w)
        xmin=0
        xmax=target_w
        ymin=0
        ymax=target_h
        if random.random() < 0.5:
            x_start = random.randint(0,(xmax-xmin)//3+xmin)
            x_end=random.randint(5*xmax//6,imgs[0].shape[1]-1)
        else:
            x_start = random.randint(0, xmin)
            x_end = random.randint(xmin+(xmax-xmin)*2//3,xmax)
        if random.random()<0.5:
            y_start=random.randint(0,(ymax-ymin)//3+ymin)
            y_end=random.randint(5*ymax//6,imgs[0].shape[0])
        else:
            y_start = random.randint(0, ymin)
            y_end = random.randint(ymin+(ymax-ymin)*2//3,ymax)

        for idx, img in enumerate(imgs):
            img = img[y_start:y_end, x_start:x_end]
            # print("shape: ",img.shape)
            imgs[idx] = img
        imgs=self.resize_images(imgs, (self.target_w, self.target_h))
        return imgs

    def random_horizontal_flip(self, imgs, rnd_flip=0.2):
        if random.random() < rnd_flip:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1).copy()
        return imgs

    def random_rotate(self, imgs, rnd_rotate=0.3):
        max_angle = 10
        if random.random() < rnd_rotate:
            angle = random.uniform(-max_angle, max_angle)
            # print("rnd angle: ",angle)
            for i in range(len(imgs)):
                img = imgs[i]
                w, h = img.shape[:2]
                rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
                img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
                imgs[i] = img_rotation
        return imgs

    def augment_data(self, img):
        tmp_rd = random.random()
        name = ""
        if tmp_rd < 0.1:
            img = add_salt_pepper_noise(img)
            name = "_salt_"
        elif tmp_rd < 0.2:
            img = augment_img(img)
            name = "_augment_"
        elif tmp_rd < 0.3:
            img = draw_noise_for_crnn(img)
            name = "_crnn_"
        elif tmp_rd < 0.4:
            noise_typ = "white_in_black"
            img = noisy_crnn(noise_typ, img)
            name = "_white_"
        # elif tmp_rd < 0.5:
        # 	noise_typ = "gauss"
        # 	img = noisy_crnn(noise_typ, img)
        # 	name = "_gauss_"
        elif tmp_rd < 0.6:
            noise_typ = "s&p"
            img = noisy_crnn(noise_typ, img)
            name = "_sp_"
        else:
            img = filter_iaa(img)
            name = "_filter_"

        if random.random() < 0.3:
            rd = random.random()
            if rd <= 0.25:
                kind_shadow = 'left'
                img = draw_shadow_full(img, kind_shadow)
            elif rd <0.5:
                kind_shadow = 'right'
                img = draw_shadow_full(img, kind_shadow)
            elif rd <= 0.75:
                kind_shadow = 'top'
                img = draw_shadow_full(img, kind_shadow)
            else:
                kind_shadow = 'bottom'
                img = draw_shadow_full(img, kind_shadow)
        # print(name)
        return img.astype(np.uint8)


class SEMValDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform = transforms.ToTensor()

        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.image_paths.sort()

    def get_basename(self, index):
        basename = os.path.basename(self.image_paths[index]).split(".")[0]
        return basename

    def __getitem__(self, index):
        image = val_resize_crop_padding(self.image_paths[index])

        # normalize image data
        image = normalize_image(image)

        return self.transform(image)

    def __len__(self):
        return len(self.image_paths)


class SEMValDataset_an_img(Dataset):
    def __init__(self, img, img_path=None):
        """
        Args:
            image_dir (str): the path where the image is located
            label_dir (str): the path where the mask is located
            transform_generator (str): transform the input image
        """
        self.transform = transforms.ToTensor()
        self.extend = ['.jpg', '.JPG', '.png', '.PNG', '.jpeg', '.JPEG']
        # self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.img = img
        self.img_path = img_path
        self.target_w = 640#1536
        self.target_h = 480#1152
        self.ratio_w = None
        self.ratio_h = None
        # self.image_paths.sort()

    def get_basename(self):
        basename = os.path.basename(self.image_path).split(".")[0]
        return basename

    def get_org_img(self):
        # image = val_resize_crop_padding(self.image_path)

        return self.img

    def __getitem__(self, index):
        image = self.img.copy()
        self.ratio_w = image.shape[1] / self.target_w
        self.ratio_h = image.shape[0] / self.target_h
        image = cv2.resize(image, (self.target_w, self.target_h))

        # image = one_resize_crop_padding(image)

        # normalize image data
        image = normalize_image(image)
        return self.transform(image)

    def __len__(self):
        return 1
