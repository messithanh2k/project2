import torch
import torch.nn.functional as F


def bce_loss(outputs, labels, confs, reduction="mean", use_focal_loss=False, device=None):
    # loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction=reduction)
    # return loss

    # mask for labeled pixel
    batch_size = labels.shape[0]
    loss = F.log_softmax(outputs, dim=1)
    # origin_loss = -loss.detach().clone()
    one_hot_label = torch.nn.functional.one_hot(labels)
    # print(one_hot_label)
    one_hot_label = one_hot_label.permute(0, 3, 1, 2)
    # print("Shapeeeeee: ", one_hot_label.shape,loss.shape)
    loss = loss * one_hot_label
    loss = torch.sum(loss, dim=1)
    negative_loss = torch.where(labels == 0, loss, torch.zeros_like(loss))
    # print(negative_loss)
    pos_loss = torch.where(labels == 1, loss, torch.zeros_like(loss))
    pos_score = pos_loss.data.exp()
    focal_loss_weight = torch.pow(1 - pos_score, 2)
    # mask_pos_weight=focal_loss_weight*labels
    # loss_nn   = nn.CrossEntropyLoss(reduce=False,size_average=False)
    # loss=loss_nn(outputs, labels)
    # origin_loss=loss.detach().clone()
    # loss*=mask_pos_weight
    pos_loss = -1. * pos_loss * focal_loss_weight
    # neg_score=negative_loss.data.exp()
    # focal_loss_weight_neg=torch.pow(1-neg_score,2)
    # negative_loss=-1.*negative_loss*focal_loss_weight_neg
    negative_loss = -3. * negative_loss

    # OHNM
    neg_labels = torch.eq(labels, 0)
    pos_labels = torch.eq(labels, 1)
    neg_losses = []
    neg_nb_pixels = []
    for idx in range(batch_size):
        neg_label = neg_labels[idx]
        pos_label = pos_labels[idx]
        cls_neg_loss = negative_loss[idx]
        n_pos = torch.sum(pos_label)
        neg_loss, nb_pixel = OHNM_single_image(n_pos, neg_label, cls_neg_loss, device)
        neg_losses.append(neg_loss)
        neg_nb_pixels.append(nb_pixel)
    neg_loss = torch.sum(torch.stack(neg_losses))
    neg_pxl = torch.sum(torch.stack(neg_nb_pixels))
    pos_loss = torch.sum(pos_loss)

    # mean_neg_loss=neg_loss/neg_pxl
    # mean_pos_loss=torch.sum(pos_loss)/torch.sum(labels)
    # final_loss=mean_pos_loss+mean_neg_loss
    # print("total loss: {:.3f}, pos loss: {:.3f}, neg loss: {:.3f},num neg pixel: {}  ".
    #       format(final_loss,mean_pos_loss.item(), mean_neg_loss.item(), neg_pxl.item()))

    total_loss = neg_loss + pos_loss
    total_pixel = neg_pxl + torch.sum(labels)
    final_loss = total_loss / total_pixel
    # print("loss: ", final_loss.item(), total_pixel.item(), neg_pxl)

    # return mean_pos_loss+mean_neg_loss
    return final_loss


def OHNM_single_image(n_pos, neg_label, cls_loss, device):
    def has_pos():
        return torch.tensor(n_pos * 3, dtype=torch.float32, device=device)

    def no_pos():
        return torch.tensor(1000, dtype=torch.float32, device=device)

    n_neg = has_pos() if n_pos > 0 else no_pos()
    max_neg_entries = torch.sum(neg_label.int())
    n_neg = n_neg if n_neg < max_neg_entries else max_neg_entries
    n_neg = n_neg.int()
    top_k_highest_loss = 1000
    thresh_loss = 0.

    def has_neg():
        neg_cls_loss = cls_loss  # torch.where(neg_label, cls_loss, torch.zeros_like(cls_loss))
        # mask_cls_loss_greater = torch.gt(cls_loss, 0.1)
        mask_neg_cls_loss_greater = torch.gt(neg_cls_loss,
                                             thresh_loss)  # torch.logical_and(mask_cls_loss_greater, neg_label)
        neg_loss_greater = torch.where(mask_neg_cls_loss_greater, neg_cls_loss, torch.zeros_like(neg_cls_loss))
        number_neg_pixel = torch.sum(mask_neg_cls_loss_greater.int())

        def has_greater():
            reduce_sum = torch.sum(neg_loss_greater)
            return reduce_sum, number_neg_pixel.float()

        def no_greater():
            return torch.max(neg_cls_loss), torch.tensor(1, dtype=torch.float32)
            # top_k=max(n_neg.item(),top_k_highest_loss)
            # total_loss_top_k=torch.sum(torch.topk(neg_cls_loss.view(-1),top_k)[0])
            # return total_loss_top_k,torch.tensor(top_k,dtype=torch.int)

        if number_neg_pixel > 0:
            reduce_sum, number_neg_pixel = has_greater()
        else:
            reduce_sum, number_neg_pixel = no_greater()
        return reduce_sum, number_neg_pixel

    def no_neg():
        return torch.tensor(0, dtype=torch.float32), torch.tensor(1, dtype=torch.float32)

    if n_neg > 0:
        reduce_sum, number_neg_pixel = has_neg()
    else:
        reduce_sum, number_neg_pixel = no_neg()

    # print("losssssssss: {:.5f}, {:.5f} ".format(reduce_sum.item(),number_neg_pixel.item()))
    return reduce_sum.to(device), number_neg_pixel.to(device)


def cross_entropy_loss(logits, labels):
    loss = F.cross_entropy(logits, labels, reduce='mean')
    return loss


def new_loss(outputs, labels):
    logpt = -F.binary_cross_entropy_with_logits(outputs, labels, reduction="none")
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt) ** 2) * logpt
    return loss.mean()


def dice_loss(outputs, labels, reduction="mean"):
    smooth = 1e-7

    b, c, h, w = outputs.size()
    loss = []
    for i in range(c):
        o_flat = outputs[:, i, :, :].contiguous().view(-1)
        l_flat = labels[:, i, :, :].contiguous().view(-1)
        intersection = torch.sum(o_flat * l_flat)
        union = torch.sum(o_flat * o_flat) + torch.sum(l_flat * l_flat)

        loss.append(1 - (2. * intersection + smooth) / (union + smooth))

    loss = torch.stack(loss)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("reduction must be either mean or sum")


def dbce_loss(outputs, labels, reduction="mean"):
    """Dice loss + BCE loss"""
    # mask for labeled pixel
    bce_loss2 = bce_loss(outputs, labels, reduction=reduction)
    dice_loss2 = dice_loss(outputs, labels, reduction=reduction)

    return (bce_loss2 + 2 * dice_loss2) / 3.0
