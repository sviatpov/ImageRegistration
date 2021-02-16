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


if __name__ == "__main__":
    tar, src = create_source_target_images()
    tar, src = tar / np.max(tar) * 255., src / np.max(src) * 255.
    cv2.imwrite("data/simple/tar.png", np.array(tar, dtype='uint8'))
    cv2.imwrite("data/simple/src.png", np.array(src, dtype='uint8'))