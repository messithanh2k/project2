import imgaug.augmenters as iaa
from scipy import ndimage
from PIL import Image, ImageFilter

import numpy as np
import random
import cv2
import os
from shutil import copyfile

KIND_TEXT = 'text'


def gen_arr_duplicate(len_a, num_dup):
    a = np.zeros(len_a)
    # print("num dup: ", num_dup)
    for i in range(len_a):
        a[i] = i // num_dup
    return a


def draw_shadow_full(img, kind_shadow):
    ny, nx, _ = img.shape
    img_fake = img.copy()
    img_fake = img_fake.astype(np.float64)

    if kind_shadow == 'left' or kind_shadow == 'right':
        nx = random.randint(30, nx // 3)

        duplicate = False
        if random.random() > 0.6:
            duplicate = True

        if duplicate:
            num_dup = 3
            if random.random() > 0.5:
                num_dup = 2

            a = gen_arr_duplicate(nx, num_dup)

            if random.random() > 0.4 and nx < 150:
                a = a * 3
            elif random.random() > 0.3 and nx < 150:
                a = a ** 2
        else:
            a = np.arange(0, nx)

            if random.random() > 0.4 and nx < 70:
                a = a * 3
            elif random.random() > 0.5 and nx < 50:
                a = a ** 2
        if kind_shadow == 'left':
            a = np.flip(a, axis=0)
        a[a > 150] = 150
        a = [a] * ny
        gradient = np.reshape(a, (len(a), len(a[0])))

        # nx = gradient.shape[1]
        if kind_shadow == 'left':
            img_fake[:ny, :nx] -= gradient[:, :, np.newaxis]
        elif kind_shadow == 'right':
            img_fake[:ny, img_fake.shape[1] - nx:] -= gradient[:, :, np.newaxis]
    elif kind_shadow == 'top' or kind_shadow == 'bottom':
        # if ny > 30:
        #     ny = random.randint(30, ny)
        # else:
        #     ny = random.randint(ny//4, ny)

        ny = random.randint(30, ny // 4)

        duplicate = False
        if random.random() > 0.6:
            duplicate = True

        if duplicate:
            num_dup = 2
            if random.random() > 0.5:
                num_dup = 3
            a = gen_arr_duplicate(ny, num_dup)  # Tao 1 array co do dai bang 1/num_dup cua shape

        # if random.random() > 0.4 and ny // num_dup < 80:
        #     a = a * 3
        # elif random.random() > 0.3 and ny // num_dup < 80:
        #     a = a ** 2
        else:
            a = np.arange(0, ny)

        # if random.random() > 0.4 and ny < 70:
        #     a = a * 3
        # elif random.random() > 0.5 and ny < 50:
        #     a = a ** 2
        a[a > 150] = 150
        if kind_shadow == 'top':
            a = np.flip(a, axis=0)

        a = [a] * nx
        gradient = np.reshape(a, (len(a), len(a[0])))
        gradient = gradient.T

        # ny, nx = gradient.shape
        if kind_shadow == 'top':
            img_fake[:ny, :nx] -= gradient[:, :, np.newaxis]
        elif kind_shadow == 'bottom':
            img_fake[img_fake.shape[0] - ny:, :nx] -= gradient[:, :, np.newaxis]
    # img_fake -= np.min(img_fake)
    # img_fake /= np.max(img_fake)
    # img_fake *= 255
    # idx_0=img_fake<0
    # idx_255=img_fake>255
    # img_fake[idx_0] = img[idx_0]
    # img_fake[idx_255] = img[idx_255]

    img_fake[img_fake < 0] = 0
    img_fake[img_fake > 255] = 255

    return img_fake.astype(np.uint8)


def noise_blur(img):
    # if img.shape[0] > 50:
    #     filter_size = random.choice(LIST_FILTER_SIZE)
    # else:
    #     filter_size = random.choice(LIST_FILTER_SIZE[: -5])

    filter_size = random.choice([15, 25])

    img = cv2.GaussianBlur(img, (filter_size, filter_size), 0)

    return img


def noisy_dotted_line(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    :param noise_typ:
    :param image:
    :return:
    """
    if noise_typ is None:
        return image.copy()
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        # var = 10
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        # amount = 0.004
        out = np.copy(image)

        # Pepper mode
        pepper_amount = 0.001
        num_pepper = np.ceil(pepper_amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0

        # Salt mode
        salt_amount = 0.01
        num_salt = np.ceil(salt_amount * image.size * s_vs_p)
        # num_salt = np.ceil(salt_amount * row * col * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        # out[coords] = 1
        out[tuple(coords)] = 255
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    elif noise_typ == "white_in_black":
        row, col, ch = image.shape
        out = image.copy()
        for r in range(row):
            for c in range(col):
                if (out[r, c] <= 100).all():
                    # or ((out[r, c] <= 100).all() and out[r, c, 0] == out[r, c, 1] and out[r, c, 1] == out[r, c, 2]):
                    if random.random() <= 0.02:
                        # out[r, c] = WHITE
                        out[r, c] = [random.randint(250, 255)] * 3
                        if r != 0 and r != row - 1 and c != 0 and c != col - 1 and random.random() <= 0.5:
                            out[r + random.choice([-1, 0, 1]), c + random.choice([-1, 0, 1])] = [random.randint(250,
                                                                                                                255)] * 3
        return out
    else:
        raise Exception("noise_typ:{} is wrong!".format(noise_typ))


def draw_noise_for_crnn(img):
    if random.random() > 0.4:
        noise_type = 'gauss'
    # elif random.random() > 0.1:
    #     noise_type = 'poisson'
    else:
        noise_type = None

    img = noisy_dotted_line(noise_type, img)

    flag_noise = False
    if random.random() > 0.3:
        img = noisy_dotted_line("s&p", img)
        flag_noise = True

    if noise_type or flag_noise and random.random() < 0.8:
        img = noise_blur(img)

    return img


def augment_jpeg(img):
    # jpeg_degree = random.choice(range(101))
    jpeg_degree = 90
    # print ("Jpeg augmenting", jpeg_degree)
    jpeg_augment = iaa.JpegCompression(jpeg_degree)
    img = jpeg_augment.augment_image(img)

    return img


def augment_img(img):
    tmp_random = random.random()
    if tmp_random < 0.2:
        img = noise_blur(img)
    elif tmp_random < 0.5:
        img = noise_blur(img)
        img = speckle(img)
    elif tmp_random < 0.8:
        img = speckle(img)
        img = noise_blur(img)
    else:
        img = speckle(img)
    # else:
    #     img = augment_jpeg(img)

    return img


def speckle(img):
    severity = np.random.uniform(0, 0.1 * 255)
    blur = ndimage.gaussian_filter(np.random.randn(*img.shape) * severity, 1)
    img_speck = (img + blur)
    img_speck[img_speck > 255] = 255
    img_speck[img_speck <= 0] = 0
    return img_speck


def add_salt_pepper_noise(img, value=255):
    img_copy = img.copy()
    ratio = random.uniform(0.005, 0.01)
    row, col, _ = img_copy.shape
    num_salt = np.ceil(img_copy.size * ratio)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    img[coords[0], coords[1], :] = value
    return img


def noisy_crnn(noise_typ, image):
    """
    Parameters
    ----------
    image : ndarray
        Input image data. Will be converted to float.
    mode : str
    One of the following strings, selecting the type of noise to add:

    'gauss'     Gaussian-distributed additive noise.
    'poisson'   Poisson-distributed noise generated from the data.
    's&p'       Replaces random pixels with 0 or 1.
    'speckle'   Multiplicative noise using out = image + n*image,where
                n is uniform noise with specified mean & variance.
    :param noise_typ:
    :param image:
    :return:
    """
    if noise_typ is None:
        return image.copy()
    if noise_typ == "gauss":
        # row, col, ch = image.shape
        # mean = 0
        # # var = 0.1
        # var = 2
        # sigma = var ** 0.01
        # gauss = np.random.normal(mean, sigma, (row, col, ch))
        # gauss = gauss.reshape(row, col, ch)
        # noisy = image + gauss
        # return noisy
        return image
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        # amount = 0.004
        out = np.copy(image)

        # Pepper mode
        pepper_amount = 0.001
        num_pepper = np.ceil(pepper_amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0

        # Salt mode
        salt_amount = 0.01
        num_salt = np.ceil(salt_amount * image.size * s_vs_p)
        # num_salt = np.ceil(salt_amount * row * col * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        # out[coords] = 1
        out[tuple(coords)] = 255
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy

    elif noise_typ == "white_in_black":
        row, col, ch = image.shape
        out = image.copy()
        for r in range(row):
            for c in range(col):
                if (out[r, c] <= 100).all():
                    # or ((out[r, c] <= 100).all() and out[r, c, 0] == out[r, c, 1] and out[r, c, 1] == out[r, c, 2]):
                    if random.random() <= 0.02:
                        # out[r, c] = WHITE
                        out[r, c] = [random.randint(250, 255)] * 3
                        if r != 0 and r != row - 1 and c != 0 and c != col - 1 and random.random() <= 0.5:
                            out[r + random.choice([-1, 0, 1]), c + random.choice([-1, 0, 1])] = [random.randint(250,
                                                                                                                255)] * 3
        return out
    else:
        raise Exception("noise_typ:{} is wrong!".format(noise_typ))


def filter_iaa(img):
    LIST_AUGMENT_FILTER = [
        # iaa.Invert(0.5),
        # iaa.AdditivePoissonNoise(scale=(0, 40)),
        iaa.AdditivePoissonNoise(10),
        iaa.AddElementwise((-10, 10), per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        iaa.AdditiveGaussianNoise(scale=0.05 * 255, per_channel=0.5), iaa.Multiply((0.5, 1.5)),
        iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255)),
        iaa.Pepper(0.01),
        # iaa.ContrastNormalization((0.5, 0.75)),\
        iaa.Add((-10, 20)),
        iaa.GammaContrast(gamma=2),
        iaa.Add((-10, 10), per_channel=0.5),
        # iaa.Affine(scale=(0.5, 1), cval=(0, 255)),
        # iaa.Affine(translate_percent={"x": (0, 0.3), "y": (0, 0.3)}),
        iaa.WithChannels(0, iaa.Add((-10, 10))), iaa.WithChannels(0, iaa.Add((-10, 10))),
        iaa.Grayscale(alpha=(0.0, 0.5)),
        iaa.GaussianBlur(sigma=(0.1, 0.3)),
    ]
    img = np.array(img, dtype=np.uint8)
    seq = iaa.SomeOf(1, LIST_AUGMENT_FILTER, random_order=True)
    image_aug, bbs_aug = seq(image=img, bounding_boxes=[])
    return image_aug


def rotate_img(img):
    h = img.shape[0]
    w = img.shape[1]
    angle = np.random.randint(3, 5)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - center[0]
    M[1, 2] += (nH / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, \
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(250, 240, 238))

    return rotated


def random_crop(img):
    h, w = img.shape[:2]
    rnd_scale = random.uniform(0.75, 1.25)
    rnd_x = random.randint(0, w)
    if rnd_x < w // 2:
        x_start = rnd_x
        x_end = w - 1
    else:
        x_start = 0
        x_end = rnd_x
    rnd_y = random.randint(0, h)
    if rnd_y < h // 2:
        y_start = rnd_y
        y_end = h - 1
    else:
        y_start = 0
        y_end = rnd_y
    new_img = img[y_start:y_end, x_start:x_end]

    # new_img=cv2.resize(new_img,(int(new_w*rnd_scale),int(new_h*rnd_scale)))

    return new_img


def main(img):
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
        elif 0.25 < rd <= 0.25:
            kind_shadow = 'right'
            img = draw_shadow_full(img, kind_shadow)
        elif 0.5 < rd <= 0.75:
            kind_shadow = 'top'
            img = draw_shadow_full(img, kind_shadow)
        else:
            kind_shadow = 'bottom'
            img = draw_shadow_full(img, kind_shadow)
    # print(name)
    return img.astype(np.uint8)


def augment_images(path_in, path_out, path_gt_org=None, path_gt_new=None):
    """
    Sinh chu bt.
    1 px = 0.75 point; 1 point = 1.333333 px
    :param table:
    :param invert:
    :param rotate:
    :param Text text_obj:
    :param max_len:
    :param utils.font_util.ListFonts list_fonts:
    :param w:
    :param h:
    :return:
    """

    files = os.listdir(path_in)

    for fn in files:
        # print(fn)
        # os.path.join(input_dir, fn)
        img = Image.open(os.path.join(path_in, fn))
        img = np.array(img)

        name = str(fn.split(".")[0])
        out = path_out
        # out = os.path.join(path_out, name)
        # if not os.path.isdir(out):
        #     os.makedirs(out)
        img_copy = img

        img = Image.open(os.path.join(path_in, fn))
        img = np.array(img)
        name = ""
        rnd_numbers = random.randint(2, 2)
        for i in range(rnd_numbers):
            if i == 0:
                tmp_rd = random.random()
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
                elif tmp_rd < 0.5:
                    noise_typ = "gauss"
                    img = noisy_crnn(noise_typ, img)
                    name = "_gauss_"
                elif tmp_rd < 0.6:
                    noise_typ = "s&p"
                    img = noisy_crnn(noise_typ, img)
                    name = "_sp_"
                else:
                    img = filter_iaa(img)
                    name = "_filter_"

                # if random.random() < 0.2:
                #     noise_typ = "speckle"
                #     img = noisy_crnn(noise_typ, img)

                if random.random() < 0.3:
                    rd = random.random()
                    if rd <= 0.25:
                        kind_shadow = 'left'
                        img = draw_shadow_full(img, kind_shadow)
                    elif 0.25 < rd <= 0.25:
                        kind_shadow = 'right'
                        img = draw_shadow_full(img, kind_shadow)
                    elif 0.5 < rd <= 0.75:
                        kind_shadow = 'top'
                        img = draw_shadow_full(img, kind_shadow)
                    else:
                        kind_shadow = 'bottom'
                        img = draw_shadow_full(img, kind_shadow)
                img = img.astype(np.uint8)
            name_f = str(fn.split(".")[0]) + name + "_" + str(i) + ".jpg"
            if path_gt_org is not None:
                gt_fname = "gt_" + fn.replace("jpg", "txt")
                new_fname = "gt_" + name_f.replace("jpg", "txt")
                gt_org = os.path.join(path_gt_org, gt_fname)
                gt_new = os.path.join(path_gt_new, new_fname)
                copyfile(gt_org, gt_new)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # if random.random()<0.5:
            #     img=rotate_img(img)
            # if random.random()<0.3:
            #     img=random_crop(img)
            # if random.random()<0.2:
            #     img=rotate_img(img)

            # expand = random.randint(50, 100)
            # new_h, new_w = img.shape[:2]
            # expand_img = np.zeros((new_h + 2 * expand, new_w + 2 * expand, 3), dtype=np.uint8)
            # std = random.randint(-5, 5)
            # b, g, r = 250, 240, 238
            # expand_img = np.full_like(expand_img, (b + std, g + std, r + std))
            # expand_img[expand:-expand, expand:-expand] = img
            cv2.imwrite(os.path.join(out, name_f), img)

            img = img_copy


# if augment and not is_black:
#     tmp_random = random.random()
#     if tmp_random < 0.2:
#         mean = int(np.mean(img))
#         if mean <= 127:
#             value = max(mean - 20, 0)
#         else:
#             value = min(mean + 20, 255)
#         img = add_salt_pepper_noise(img, value)
#     # fn=str(random.randint(0,1000))+".jpg"
#     # cv2.imwrite("debug/"+fn,img)
#     elif tmp_random < 0.4:
#         img = noise_blur(img)
#     elif tmp_random < 0.5:
#         img = noise_blur(img)
#         img = speckle(img)
#     elif tmp_random < 0.6:
#         img = speckle(img)
#         img = noise_blur(img)
#     elif tmp_random < 0.8:
#         img = speckle(img)
#     else:
#         img = augment_jpeg(img)
# return img


if __name__ == "__main__":
    augment_images("/media/aimenext/Newdisk/cuongdx/pixellink/data/data/sci/color_1113/all/imgs",
                   "/media/aimenext/Newdisk/cuongdx/pixellink/data/data/sci/color_1113/augment/imgs",
                   "/media/aimenext/Newdisk/cuongdx/pixellink/data/data/sci/color_1113/all/gt",
                   "/media/aimenext/Newdisk/cuongdx/pixellink/data/data/sci/color_1113/augment/gt")
