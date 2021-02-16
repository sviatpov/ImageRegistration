"Simple implementation of image registration algorithm"

# Inspired by article:
#   "Parametric Image Alignment Using Enhanced
#   Correlation Coefficient Maximization"
#   Autors: Georgios Evangelidis, Emmanouil Psarakis

import numpy as np
import cv2

class ECC():

    def __init__(self, learning_rate=1.5, iter=100, Rt=cv2.getRotationMatrix2D((0,0), 0, 1.0)):
        # initial rotation
        self.rt = Rt

        # some kind of learning rate =)
        self.ln = learning_rate

        self.iter = iter

        self.method = "Translation"
        self.table_of_method = {"Euclidian" : self.jacobi_euclidian,
                                "Rotation" : self.derivation_rotation,
                                "Translation" : self.jajacobi_translation}

    def set_method(self, m):
        if m in self.table_of_method:
            self.method = m

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

    @staticmethod
    def derivation_rotation(gx, gy, rt, shape):
        """
            Image derivation along rotation angle is:

                Jangl = dI / dx * dx/dangl + dI / dy * dy / dangl

            Where dx/dangl, dy / dangl we can find from here:
                Xnew(thet, Tx, Ty) = X * cos(angl) - Y * sin(angl) + Tx
                Ynew(thet, Tx, Ty) = X * sin(angl) + Y * cos(angl) + Ty

                d(Xnew)/d(angl) = -X * sin(angl) - Y * cos(angl)
                d(Ynew)/d(angl) =  X * cos(angl) - Y * sin(angl)

        :param gx: normalized image derivation along x
        :param gy: normalized image derivation along y
        :param rt: [2x3] transformation matrix
        :param shape: image shape
        :return: Image derivation along rotation in image plane
        """

        si = rt[1, 0]
        co = rt[0, 0]

        # It is the same procedure as a commented cycle below
        x = np.arange(0, shape[1], 1, dtype='int8')
        y = np.arange(0, shape[0], 1, dtype='int8')
        xx, yy = np.meshgrid(x, y)
        Jangl = (-si * yy - co * xx) * gx + (yy * co - xx * si) * gy

        # Jangl = np.zeros_like(gx, dtype="float64")
        # for x in range(Jangl.shape[0]):
        #     for y in range(Jangl.shape[1]):
        #        Jangl[x,y] = (-si * y - co * x ) * gx[x,y]  +\
        #                     (y * co - x * si) * gy[x,y]

        return Jangl

    @staticmethod
    def derivation_tx(gx):

        """
        Image derivation along Tx:

            Jtx = dI / dx * dx/dTx + dI / dy * dy / dTx

        Where dx/dTx, dy / dTx we can find from here:
            Xnew(thet, Tx, Ty) = X * cos(angl) - Y * sin(angl) + Tx
            Ynew(thet, Tx, Ty) = X * sin(angl) + Y * cos(angl) + Ty

            d(Xnew)/d(Tx) = 1
            d(Ynew)/d(Tx) = 0
        :param gx: normalized image derivation along x
        :return: image derivation along Tx
        """
        return gx

    @staticmethod
    def derivation_ty(gy):

        """
        Image derivation along Ty:

            Jtx = dI / dx * dx/dTy + dI / dy * dy / dTy

        Where dx/dTy, dy / dTy we can find from here:
            Xnew(thet, Tx, Ty) = X * cos(angl) - Y * sin(angl) + Tx
            Ynew(thet, Tx, Ty) = X * sin(angl) + Y * cos(angl) + Ty

            d(Xnew)/d(Ty) = 0
            d(Ynew)/d(Ty) = 1
        :param gy: normalized image derivation along y
        :return: image derivation along Ty
        """
        return gy

    @staticmethod
    def jajacobi_translation(gx, gy, rt=None, shape=None):
        """
        Assume 2 degree of freedom, translation along Ox and Oy
        :param gx:
        :param gy:
        :return:
        """
        return ECC.derivation_tx(gx), ECC.derivation_ty(gy)

    @staticmethod
    def jacobi_euclidian(gx, gy, rt, shape):
        """
        Assume that we have 3-degree of freedom: rotation in image plane and translation along Ox and Oy,
        so jacobian contains 3 parts.
        :param gx: derivation along x axis
        :param gy: derivation along y axis
        :param rt: transformation matrix
        :param shape: image shape
        :return:
        """
        Dtx, Dty = ECC.jajacobi_translation(gx, gy)
        Drot = ECC.derivation_rotation(gx, gy, rt, shape)
        return Drot, Dtx, Dty

    @staticmethod
    def multiplyJacobi(jac):
        """
            sum(J1*J1) sum(J1*J2)
        H = sum(J2*J1) sum(J2*J2)
        :param jac:
        :return:
        """
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

    @staticmethod
    def update(rt, dtheta=0, dtx=0, dty=0):
        """
        :param rt: [2x3] [R | T] Matrix
        :param drt: [1x3] updates for theta, x, y
        :return:
        """
        theta = np.arcsin(rt[1,0])
        theta = theta + dtheta
        rt[0,0] = np.cos(theta)
        rt[0,1] = -np.sin(theta)
        rt[0,2] = rt[0,2] + dtx

        rt[1,0] = np.sin(theta)
        rt[1,1] = np.cos(theta)
        rt[1,2] = rt[1,2] + dty

    def align(self, src, tar):

        # [X, Y] - pixel on image
        # Xnew = X * cos - Y * sin + Tx
        # Ynew = X * sin + Y * cos + ty

        rt = np.copy(self.rt)
        calculate_jacobian = self.table_of_method.get(self.method)
        g = src
        gy = cv2.Sobel(g, cv2.CV_16SC1, 0, 1, ksize=5, borderType=cv2.BORDER_REFLECT)
        gx = cv2.Sobel(g, cv2.CV_16SC1, 1, 0, ksize=5, borderType=cv2.BORDER_REFLECT)

        for i in range(100):
            ## Warp source image
            ## [R | T] @ Im, It means:
            ##  Imnew[Xnew, Ynew] = Im[X, Y]
            gw =  cv2.warpAffine(g, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)

            # for debug
            save(gw - tar, "data/simple/res_{}.png".format(i))

            ## The same for image derivations
            gxw = cv2.warpAffine(gx, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)
            gyw = cv2.warpAffine(gy, rt, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR)


            g_mean, g_std = np.mean(gw), np.std(gw)
            t_mean, t_std = np.mean(tar), np.std(tar)
            ratio_std = g_std / t_std


            gxw, gyw = gxw/np.max(gxw), gyw/np.max(gyw)
            Jacc = calculate_jacobian(gxw, gyw, rt, shape=(g.shape[0], g.shape[1]))
            if type(Jacc) == np.ndarray:
                Jacc = [Jacc]
            Hess = self.multiplyJacobi(Jacc)

            e = tar * (-ratio_std) + gw
            e = e - (g_mean - ratio_std * t_mean)

            ep = self.project_onto_jacobi(e, Jacc)

            drt = (np.linalg.inv(Hess) @ ep * (self.ln))[:, 0]

            # TODO: fixme
            if self.method == "Euclidian":
                self.update(rt, dtheta=drt[0], dtx=drt[1], dty=drt[2])
            elif self.method == "Translation":
                self.update(rt, dtx=drt[0], dty=drt[1])
            elif self.method == "Rotation":
                self.update(rt, dtheta=drt[0][0])

def save(im, path):
    im = np.abs(im)
    im = np.array(im, dtype='float64') / np.max(im) * 255.
    cv2.imwrite(path, np.array(im, dtype='uint8'))
if __name__ == "__main__":

    source = cv2.imread("data/simple/src.png", cv2.IMREAD_GRAYSCALE)
    target = cv2.imread("data/simple/tar.png", cv2.IMREAD_GRAYSCALE)

    source = np.array(source, dtype='float64') / 255.
    target = np.array(target, dtype='float64') / 255.

    ecc = ECC()
    ecc.align(source, target)
    # for im in ecc.pyramid_tar:
    #     normilize_image(im, "/home/svipov/Documents/projects/ECC/data/res_")

    # cv2.imshow("dv", normilize_image(pattern))
    # cv2.imshow("sdf", normilize_image(warped-pattern))
    # cv2.waitKey()

