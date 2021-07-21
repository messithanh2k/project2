from scipy.spatial import distance as dist
from imutils import perspective, contours
import numpy as np
import imutils
import cv2
from predict_image import predict, val_resize_crop_padding

DICT = {0: 'background', 1: 'mouse', 2: 'phone', 3: 'book', 4: 'ruler'}
width = 20


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def get_rec(box):
    x_min, y_min = np.min(box, axis=0)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max, y_max = np.max(box, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)


def measure(image):
    image = val_resize_crop_padding(image)
    inp = predict(image)
    gray_ = np.where(inp == 0, inp, 255)
    gray = cv2.GaussianBlur(gray_, (7, 7), 0)

    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # sort the contours from left-to-right and initialize the
    # 'pixels per metric' calibration variable
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    # loop over the contours individually
    class_list = []
    distance_list = []
    orig_list = []
    coord_list = []
    orig = image.copy()
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 1500:
            continue
        # compute the rotated bounding box of the contour
        # orig = image.copy()
        box = cv2.minAreaRect(c)
        if imutils.is_cv2():
            box = cv2.cv.BoxPoints(box)
        else:
            box = cv2.boxPoints(box)
        box = np.array(box, dtype="int")
        # order the points in the contour such that they appear
        # in top-left, top-right, bottom-right, and bottom-left
        # order, then draw the outline of the rotated bounding
        # box
        box = perspective.order_points(box)
        wrap = perspective.four_point_transform(inp, box)
        x_min, y_min, x_max, y_max = get_rec(box)

        classes = np.unique(wrap, return_counts=True)
        index = np.argmax(classes[1])
        class_id = classes[0][index]
        class_list.append(class_id)
        if class_id == 3:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.1 * peri, True)
            approx = np.reshape(approx, (-1, 2))
            if len(approx) == 4:
                box = perspective.order_points(approx)
        cv2.drawContours(orig, [box.astype("int")], -1, (255, 255, 0), 2)
        # loop over the original points and draw them
        for (x, y) in box:
            cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(orig, DICT[class_id],
                    (x_min + 5, y_min + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2)
        # unpack the ordered bounding box, then compute the midpoint
        # between the top-left and top-right coordinates, followed by
        # the midpoint between bottom-left and bottom-right coordinates
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # compute the midpoint between the top-left and top-right points,
        # followed by the midpoint between the top-righ and bottom-right
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        coord_list.append((tltrX, tltrY, trbrX, trbrY))
        # draw the midpoints on the image
        cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        # draw lines between the midpoints
        cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
        cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

        # orig_list.append(orig)
        # compute the Euclidean distance between the midpoints
        dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
        dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

        distance_list.append((dA, dB))
        # if the pixels per metric has not been initialized, then
        # compute it as the ratio of pixels to supplied metric
        # (in this case, inches)
        if pixelsPerMetric is None and class_id == 2:
            pixelsPerMetric = max(dA, dB) / width

    for i in range(len(class_list)):
        if pixelsPerMetric is not None:
            dimA = distance_list[i][0] / pixelsPerMetric
            dimB = distance_list[i][1] / pixelsPerMetric
            cv2.putText(orig, "{:.1f}cm".format(dimA),
                        (int(coord_list[i][0] + 10), int(coord_list[i][1] + 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
            cv2.putText(orig, "{:.1f}cm".format(dimB),
                        (int(coord_list[i][2] - 50), int(coord_list[i][3])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
    return orig


vid = cv2.VideoCapture("video/VID_20210528_160154_3.mp4")
if not vid.isOpened():
    raise IOError("Couldn't open webcam or video")
video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
video_fps = vid.get(cv2.CAP_PROP_FPS)
video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

result = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'DIVX'), 20, video_size)

i = 0
while True:
    ret, frame = vid.read()
    if ret:
        i = i + 1
        frame = measure(frame)
        cv2.imwrite("image_cut/project/%s.jpg" % str(i), frame)
        # result.write(frame)
    else:
        break

# Release everything if job is finished
vid.release()
# result.release()
cv2.destroyAllWindows()


