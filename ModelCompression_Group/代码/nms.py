import numpy as np
import cv2

def non_max_suppression_slow(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in xrange(0, last):
            j = idxs[pos]

            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)
    return boxes[pick]

def main():
    images = [
        ("/home/longrj/plankton/models/nms/timg1.jpg", np.array([
            (12, 84, 140, 212),
            (24, 84, 152, 212),
            (36, 84, 164, 212),
            (12, 96, 140, 224),
            (24, 96, 152, 224),
            (24, 108, 152, 236)])),
        ("/home/longrj/plankton/models/nms/timg2.jpg", np.array([
            (114, 60, 178, 124),
            (120, 60, 184, 124),
            (114, 66, 178, 130)])),
        ("/home/longrj/plankton/models/nms/timg3.jpg", np.array([
            (12, 30, 76, 94),
            (12, 36, 76, 100),
            (72, 36, 200, 164),
            (84, 48, 212, 176)]))]
    for (imagePath, boundingBoxes) in images:
        print "[x] %d initial bounding boxes" % (len(boundingBoxes))
        image = cv2.imread(imagePath,0)
        orig = image.copy()

        for (startX, startY, endX, endY) in boundingBoxes:
            cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 255), 2)

        pick = non_max_suppression_slow(boundingBoxes, 0.3)
        print "[x] after applying non-maximum, %d bounding boxes" % (len(pick))

        for (startX, startY, endX, endY) in pick:
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)

        #cv2.imshow("Original", orig)
        #cv2.imshow("After NMS", image)
        #cv2.waitKey(0)

if __name__=='__main__':
    main()