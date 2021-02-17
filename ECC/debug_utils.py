import numpy as np
import cv2

### FOR DEBUG

def create_source_target_images():
    shape = (400, 800)
    pattern = np.zeros(shape=shape, dtype="float64")
    pattern[int(shape[0] / 2 - 100): int(shape[0] / 2 + 100), int(shape[1] / 2 - 100): int(shape[1] / 2 + 100)] = 1.
    pattern[int(shape[0] / 2 - 10): int(shape[0] / 2 + 10), int(shape[1] / 2 - 10): int(shape[1] / 2 + 10)] = 0.5

    image_center = tuple(np.array(pattern.shape[1::-1]) / 2 - 50)
    rt = cv2.getRotationMatrix2D(image_center, 10, 1.0)
    # rt = np.zeros(shape=(2,3))
    # rt[0,0], rt[1,1] = 1,1
    rt[0,2], rt[1,2] = 50,50
    print(rt)

    warped = cv2.warpAffine(pattern, rt, (pattern.shape[1], pattern.shape[0]), flags=cv2.INTER_LINEAR)
    return pattern, warped

def warp_image(impath):
    im = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
    im = cv2.copyMakeBorder(im, 100, 100, 100, 100, cv2.BORDER_CONSTANT, None, [0,0,0])

    print(im.shape)
    print(im.shape[1::-1])
    rt = cv2.getRotationMatrix2D(tuple(np.array(im.shape[1::-1]) / 2), 3, 1.0)

    rt[0, 2], rt[1, 2] = 10, 15
    warped = cv2.warpAffine(im, rt, tuple(np.array(im.shape[1::-1])), flags=cv2.INTER_LINEAR)

    # cv2.imshow("sdcf", im)
    # cv2.imshow("w", warped)
    # cv2.waitKey(0)
    return im, warped
if __name__ == "__main__":
    tar, src = warp_image("data/complex/eyfel.jpg")
    cv2.imwrite("data/complex/eyfel_src.png", src)
    cv2.imwrite("data/complex/eyfel_tar.png", tar)
    #
    # tar, src = create_source_target_images()
    # tar, src = tar / np.max(tar) * 255., src / np.max(src) * 255.
    # cv2.imwrite("data/simple/tar.png", np.array(tar, dtype='uint8'))
    # cv2.imwrite("data/simple/src.png", np.array(src, dtype='uint8'))