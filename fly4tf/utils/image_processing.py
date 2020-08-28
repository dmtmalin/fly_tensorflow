import cv2


def _contour_is_bad(c):
    epsilon = 5.0
    approx = cv2.approxPolyDP(c, epsilon, True)
    return len(approx) <= 3


def findContours(img, sort = True):
    im_copy = img.copy()
    gray = cv2.cvtColor(im_copy, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    _, th = cv2.threshold(blur, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if not _contour_is_bad(c)]
    if sort:
        contours = sortContours(contours)
    return contours


def sortContours(contours, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boxes = [cv2.boundingRect(c) for c in contours]
    contours, _ = zip(*sorted(zip(contours, boxes),
                              key=lambda b: b[1][i], reverse=reverse))
    return contours


def boundingRectContour(c):
    poly = cv2.approxPolyDP(c, 3, True)
    return cv2.boundingRect(poly)


def extremumContour(c):
    left = tuple(c[c[:, :, 0].argmin()][0])
    right = tuple(c[c[:, :, 0].argmax()][0])
    top = tuple(c[c[:, :, 1].argmin()][0])
    bot = tuple(c[c[:, :, 1].argmax()][0])
    return left, right, top, bot


def sliceImage(img, _h: int, _w: int):
    points = []
    size = {'h': _h, 'w': _w}
    step_x = _w
    step_y = _h
    h, w = img.shape[:2]
    p = {'x': 0, 'y': 0}
    while p['y'] + size['h'] <= h:
        while p['x'] + size['w'] <= w:
            point = (p['x'], p['y'])
            points.append(point)
            p['x'] += step_x
        p['x'] = 0
        p['y'] += step_y
    return [(img[y:y + _h, x:x + _w], x, y) for x, y in points]
