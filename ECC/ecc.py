"Simple implementation of image registration algorithm"

# Inspired by article:
#   "Parametric Image Alignment Using Enhanced
#   Correlation Coefficient Maximization"
#   Autors: Georgios Evangelidis, Emmanouil Psarakis

import numpy as np
import cv2
import os

class ECC():
    def __init__(self, src, tar):
        """
        :param src: source image
        :param tar: target image
        """
        self.src = np.copy(src)
        self.tar = np.copy(tar)
        # initial rotation
        self.rt = cv2.getRotationMatrix2D((0,0), 0, 1.0)
        # some kind of learning rate =)
        self.ln = 1.5


        # TODO: image pyramid; experimental mess
        self.pyramid_src = self.build_pyr(self.src)
        self.pyramid_tar = self.build_pyr(self.tar)

    @staticmethod
    def build_pyr(im):
        pyramid = []
        pyramid.append(
            cv2.GaussianBlur(im, (5, 5), sigmaY=0, sigmaX=0, borderType=cv2.BORDER_REFLECT))
        sz = len(im)
        while sz > 64:
            tmp = cv2.GaussianBlur(pyramid[-1], (5, 5), sigmaY=0, sigmaX=0, borderType=cv2.BORDER_REFLECT)
            tmp = tmp[::2, ::2]
            pyramid.append(tmp)
            sz = len(tmp)
        return pyramid

    def jacobi_euclidian(self, gx, gy, rt, shape):
        """
        :param gx: derivation along x axis
        :param gy: derivation along y axis
        :param rt: transformation matrix
        :param shape: image shape
        :return:
        """
        gx = gx / np.max(gx)
        gy = gy / np.max(gy)
        si = rt[1, 0]
        co = rt[0, 0]
        x = np.arange(0, shape[1], 1, dtype='int8')
        y = np.arange(0, shape[0], 1, dtype='int8')
        # It is the same procedure as a commented cycle below
        xx, yy = np.meshgrid(x, y)
        Jangl = (-si * yy- co * xx) * gx + (yy * co - xx * si) * gy
        # Jangl = np.zeros_like(gx, dtype="float64")
        # for x in range(Jangl.shape[0]):
        #     for y in range(Jangl.shape[1]):
        #        Jangl[x,y] = (-si * y - co * x ) * gx[x,y]  +\
        #                     (y * co - x * si) * gy[x,y]

        Jtx, Jty = np.copy(gx), np.copy(gy)
        return Jangl, Jtx, Jty

    def multiplyJacobi(self, jac):
        dim = len(jac)
        H = np.zeros(shape=(dim, dim), dtype="float64")
        for i in range(0, dim):
            H[i, i] = np.sum(jac[i] * jac[i])
            for j in range(i + 1, dim):
                H[i, j] = np.sum(jac[i] * jac[j])
                H[j, i] = H[i, j]

        return H


    def project_onto_jacobi(self, err, jac):
        ep = np.zeros(shape=(len(jac), 1), dtype='float64')
        for i in range(len(jac)):
            ep[i] = np.sum(err * jac[i])
        return ep

    def update(self, rt, drt):
        theta = np.arcsin(rt[1,0])
        theta = theta + drt[0]
        rt[0,0] = np.cos(theta)
        rt[0,1] = -np.sin(theta)
        rt[0,2] = rt[0,2] + drt[1]

        rt[1,0] = np.sin(theta)
        rt[1,1] = np.cos(theta)
        rt[1,2] = rt[1,2] + drt[2]

    def align(self, g):
        ## x * Cos - Y * Sin + tx
        ## X * sin + Y * cos + ty
        rt = np.copy(self.rt)
        gy = cv2.Sobel(g, cv2.CV_16SC1, 0, 1, ksize=5, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(g, cv2.CV_16SC1, 1, 0, ksize=5, borderType=cv2.BORDER_REFLECT)

        for i in range(100):
            gw =  cv2.warpAffine(g, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
            # normilize_image(gw, "/home/svipov/Documents/projects/ECC/data/res_{}.png".format(i))
            gxw = cv2.warpAffine(gx, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
            gyw = cv2.warpAffine(gy, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
            # normilize_image(gxw, "/home/svipov/Documents/projects/ECC/data/gxw_{}.png".format(i))
            # normilize_image(gyw, "/home/svipov/Documents/projects/ECC/data/gyw_{}.png".format(i))
            g_mean, g_std = np.mean(gw), np.std(gw)
            t_mean, t_std = np.mean(self.tar), np.std(self.tar)
            ratio_std = g_std / t_std

            Jacc = self.jacobi_euclidian(gxw, gyw, rt, shape=(g.shape[0], g.shape[1]))
            Hess = self.multiplyJacobi(Jacc)
            print(np.linalg.inv(Hess))
            # normilize_image(Jacc[1], "/home/svipov/Documents/projects/ECC/data/Jtx_{}.png".format(i))
            # normilize_image(gyw, "/home/svipov/Documents/projects/ECC/data/gyw_{}.png".format(i))
            e = self.tar * (-ratio_std) + gw
            e = e - (g_mean - ratio_std * t_mean)

            normilize_image(e, "/home/svipov/Documents/projects/ECC/data/err_{}.png".format(i))
            ep = self.project_onto_jacobi(e, Jacc)
            # print("ERRR: ", ep)
            Jacc = np.concatenate(Jacc, axis=0)
            # normilize_image(Jacc, "/home/svipov/Documents/projects/ECC/data/jac_{}.png".format(i))
            drt = np.linalg.inv(Hess) @ ep * (self.ln)
            # drt[0] = drt[0] * (-1)
            # print(drt)
            # drt = Hess @ ep * (-self.ln)
            self.update(rt, drt)


def normilize_image(im, save=None):
    im = np.abs(im)
    im = np.array(im, dtype='float64') / np.max(im) * 254.
    im = np.array(im, dtype='uint8')
    if save is not None:
        cv2.imwrite(save, im)
    return im

if __name__ == "__main__":

    def create_source_target_images():
        shape = (400,800)
        pattern = np.zeros(shape=shape, dtype="float64")
        pattern[int(shape[0]/2 - 100) : int(shape[0]/2 + 100), int(shape[1]/2 - 100) : int(shape[1]/2 + 100)] = 1.
        pattern[int(shape[0]/2 - 10) : int(shape[0]/2 + 10), int(shape[1]/2 - 10) : int(shape[1]/2 + 10)] = 0.5

        image_center = tuple(np.array(pattern.shape[1::-1]) / 2)
        rt = cv2.getRotationMatrix2D(image_center, 10, 1.0)
        # rt = np.zeros(shape=(2,3))
        # rt[0,0], rt[1,1] = 1,1
        # rt[0,2], rt[1,2] = 0,0
        warped = cv2.warpAffine(pattern, rt, (pattern.shape[1], pattern.shape[0]), flags=cv2.INTER_LINEAR)
        return pattern, warped

    pattern, warped = create_source_target_images()
    ecc = ECC(warped, pattern)
    ecc.align(warped)
    # for im in ecc.pyramid_tar:
    #     normilize_image(im, "/home/svipov/Documents/projects/ECC/data/res_")

    # cv2.imshow("dv", normilize_image(pattern))
    # cv2.imshow("sdf", normilize_image(warped-pattern))
    # cv2.waitKey()

