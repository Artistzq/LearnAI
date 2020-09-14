import imageio
import math
import time
from typing import List
from matplotlib import pyplot as plt
import numpy as np


class Pyramid:

    def __init__(self, image, level: int = -1, kernel=None):
        """
        :param image: 需处理的图像
        :param level: 高斯金字塔的层数（不包括原图），默认最大层数
        :param kernel: 滤波器，默认3*3均值滤波
        """
        self.image: np.ndarray = np.array(image)
        max_level = math.ceil(np.log2(min(image.shape[:2])))
        # 设置层数，默认是最大层数（到1*1图像）
        if level < 0 or level > max_level:
            level = max_level
        elif max_level >= level >= 0:
            pass
        self.level = level
        if kernel is None:
            self.kernel = np.array([[1 / 9] * 9]).reshape((3, 3))
        else:
            kernel = np.asarray(kernel)
            assert kernel.ndim == 2, "无效滤波模板"
            self.kernel = kernel
        self.gauss = []  # type: List[np.ndarray]
        self.laplace = []  # type: List[np.ndarray]
        self.rebuildings = []  # type: List[np.ndarray]

    def up_generate(self, origin_image: np.ndarray, up_size=None):
        """
        上采样图像，并进行滤波
        :param origin_image: 需要上采样的图像
        :param up_size: 上采样后的尺寸，默认为原图的两倍
        :return: 经过上采样，并通过低通滤波器的图像
        """
        if up_size is None:
            # 默认处理
            oh, ow = origin_image.shape[0], origin_image.shape[1]
            up = self.resize(origin_image, (2 * oh, 2 * ow))
        else:
            # 根据传入尺寸采样
            assert len(tuple(up_size)) == 2, "尺寸错误"
            up = self.resize(origin_image, up_size)
        kh = min(up.shape[0], self.kernel.shape[0])
        kw = min(up.shape[1], self.kernel.shape[1])
        # 防止滤波器尺寸超过图像尺寸
        if up.shape[0] < self.kernel.shape[0] or up.shape[1] < self.kernel.shape[1]:
            new_k = [1 / (kw * kh)] * (kh * kw)
            up = self.filter(up, np.array(new_k).reshape((kh, kw)))
        else:
            up = self.filter(up, self.kernel)
        return up

    def down_generate(self, origin_image: np.ndarray):
        """
        图像通过低通滤波器，并下采样，步长为2
        :param origin_image: 原图像
        :return: 经过下采样和低通滤波的图像
        """
        oh, ow = origin_image.shape[0], origin_image.shape[1]
        kh = min(oh, self.kernel.shape[0])
        kw = min(ow, self.kernel.shape[1])
        # 滤波，如果卷积核尺寸比图像小，卷积核就缩小
        if oh < self.kernel.shape[0] or ow < self.kernel.shape[1]:
            new_k = [1 / (kw * kh)] * kh * kw
            origin_image = self.filter(origin_image, np.array(new_k).reshape((kh, kw)))
        else:
            origin_image = self.filter(origin_image, self.kernel)
        # 下采样
        down = self.resize(origin_image, ((oh + 1) // 2, (ow + 1) // 2))
        return down

    def generate_gauss_pyramid(self):
        """生成高斯金字塔"""
        self.gauss.clear()
        gau = self.image.astype(np.float)
        self.gauss.append(gau)
        for i in range(self.level):
            # 向下生成
            gau = self.down_generate(gau)
            self.gauss.append(gau)

    def generate_laplace_pyramid(self):
        """生成拉普拉斯金字塔"""
        assert self.gauss, "须先进行高斯金字塔生成"
        self.laplace.clear()
        for j in range(self.level, 0, -1):
            # 上一层的高斯图像进行上采样，再做差
            lap = self.up_generate(self.gauss[j], up_size=self.gauss[j - 1].shape[:2])
            lap = self.gauss[j - 1] - lap
            self.laplace.insert(0, lap)

    def generate_pyramid(self):
        """生成图像高斯金字塔和拉普拉斯金字塔"""
        # 先生成高斯金字塔
        self.generate_gauss_pyramid()
        # 再生成拉普拉斯金字塔
        self.generate_laplace_pyramid()

    def rebuild(self):
        """根据拉普拉斯金字塔和最高层高斯图重建每一层得原图"""
        assert self.gauss and self.laplace, "须先进行金字塔生成"
        self.rebuildings.clear()
        gau = self.gauss[-1]  # 只用最高层
        # 从顶端往下重建
        for i in range(len(self.gauss) - 1, 0, -1):
            gau = self.up_generate(gau, self.gauss[i - 1].shape[:2])
            gau += self.laplace[i - 1]
            # 本次上采样、滤波、再相加拉普拉斯图像得到的高斯图像
            # 当作下一次上采样的原图片
            self.rebuildings.insert(0, gau)

    def get_output(self):
        """处理金字塔中的矩阵，使其可以以图像显示，且效果较好"""
        ret = [[], [], []]
        for num, lst in enumerate([self.gauss, self.laplace, self.rebuildings]):
            for (i, l) in enumerate(lst):
                if num == 0 or num == 2:
                    ret[num].append(self.format(l, grayscale_trans=False))
                else:
                    ret[num].append(self.format(l))
        return ret

    @classmethod
    def format(cls, image, grayscale_trans=True):
        """
        设置矩阵image，使其能够以图像形式输出
        :param image: 像素矩阵
        :param grayscale_trans: 是否进行灰度变化，默认进行线性灰度变换
        :return:
        """
        l = image
        if grayscale_trans:
            l = np.copy(image)
            if l.ndim == 2 or (l.ndim == 3 and l.shape[2] == 1):
                # 单通道灰度图
                a, b = np.min(l), np.max(l)
                if a == b:
                    l = np.clip(l, 0, 255)
                else:
                    l = (l - a) * 255 / (b - a)
            else:
                # 彩色图
                for i in range(3):
                    a, b = np.min(l[:, :, i]), np.max(l[:, :, i])
                    if a == b:
                        l[:, :, i] = np.clip(l[:, :, i], 0, 255)
                    else:
                        l[:, :, i] = (l[:, :, i] - a) * 255 / (b - a)
        else:  # 否则不进行灰度变换，直接取整再限制范围
            pass
        l = l.astype(int)
        l = np.clip(l, 0, 255).astype(np.uint8)
        return l

    @classmethod
    def filter(cls, image: np.ndarray, kernel: np.ndarray):
        """
        图像image经过kernel卷积核进行卷积操作
        :param image: 原图像
        :param kernel: 卷积核
        :return: 卷积后的图像
        """
        # 预处理
        # return cv2.filter2D(image, -1, kernel)
        I, F = np.asarray(image), np.asarray(kernel)
        assert I.ndim == 2 or I.ndim == 3, "无效图片"
        assert F.ndim == 2, "无效卷积核"
        assert F.shape[0] <= I.shape[0] and F.shape[1] <= I.shape[1], "卷积核尺寸超过图像" + str(F.shape) + ">" + str(I.shape)
        if I.ndim == 2:
            I = np.expand_dims(I, axis=-1)

        # 获取相关参数
        i_height, i_width = I.shape[0], I.shape[1]  # 原图高宽
        f_height, f_width = F.shape[0], F.shape[1]  # 卷积核高宽

        # 这里选定卷积核的中心为：
        # 对于奇数的高/宽，取中心位置，例如[1,2,3,4,5]，则3为中心
        # 对于偶数的高/宽，无中心点，则取中心靠右的点，例如[1,2,3,4], 则3为中心
        # 再如，对于[[1,2],[3,4]]，则中心为元素4
        half_f_h, half_f_w = int(
            f_height / 2), int(f_width / 2)  # 卷积核中心上部高宽，也即中心坐标

        # 卷积核中心上部top_pad，底部bottom_pad；左边left_pad，右边right_pad分别多少个像素（不包括中心位置）
        # 对于2*2的卷积核：top_pad = 1, bottom_pad = 0; left_pad = 1, right_pad = 0
        top_pad, bottom_pad = half_f_h, f_height - half_f_h - 1
        left_pad, right_pad = half_f_w, f_width - half_f_w - 1

        # 边缘镜像对称扩展，扩展范围由卷积核尺寸确定
        I_expand = np.pad(
            I, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode="symmetric")

        # 使用矩阵乘法运算加快速度, 循环内只赋值，不设乘法
        # 把卷积核展开成1-D的列向量，原图每个元素附近的切片展开成行向量
        # 进行矩阵乘法（向量内积）
        length = kernel.shape[0] * kernel.shape[1]
        kernel = kernel.flatten().reshape((length,))
        target_image = np.zeros(I.shape)
        temp = np.zeros(I.shape + (1, length))
        for num in range(I.shape[2]):
            for i in range(i_height):
                for j in range(i_width):
                    slice = I_expand[i: i + f_height, j: j + f_width, num]
                    slice = slice.flatten().reshape((1, length))
                    temp[i, j, num] = slice
            target_image[:, :, num] = temp[:, :, num, :, :].dot(kernel)[:, :, 0]

        if target_image.shape[2] == 1:
            return target_image[:, :, 0]
        return target_image

    @classmethod
    def resize(cls, origin_image, target_size):
        """
        对图像origin_image进行缩放，使其尺寸变为target_size，双线性插值

        :param origin_image: 需要缩放的图像
        :param target_size: 想要缩放的尺寸：（高，宽）
        :return: 缩放后的图像
        """
        origin_image = np.asarray(origin_image)
        assert origin_image.ndim == 3 or origin_image.ndim == 2, "只接受三通道或单通道图片"
        if origin_image.shape[:2] == target_size:  # 大小相等直接返回
            return np.copy(origin_image)

        if origin_image.ndim == 2:
            origin_image = np.expand_dims(origin_image, -1)
        osize = origin_image.shape
        target_shape = target_size + origin_image.shape[2:]
        # 原图边缘扩充，在图像周围增添一圈，值为图像边缘值
        origin_image = np.pad(origin_image, ((1, 1), (1, 1), (0, 0)), mode="edge")
        # 缩放比例 scale_width (column), scale_height (row)
        scale_h, scale_w = (
            (osize[0]) / (target_shape[0]), (osize[1]) / (target_shape[1]))
        # 矩阵乘法代替循环
        th, tw = target_shape[0], target_shape[1]
        INDEX = np.meshgrid(range(th), range(tw), indexing="ij")
        INDEX = np.array(INDEX, dtype=np.float).reshape(2, th * tw)
        INDEX += 0.5
        INDEX *= np.array([scale_h, scale_w]).reshape(2, 1)
        INDEX -= 0.5
        INDEX += 1
        INT_INDEX = INDEX.astype(np.int)
        V1 = np.reshape(origin_image[INT_INDEX[0], INT_INDEX[1]], target_shape)
        V2 = np.reshape(origin_image[INT_INDEX[0], INT_INDEX[1] + 1], target_shape)
        V3 = np.reshape(origin_image[INT_INDEX[0] + 1, INT_INDEX[1]], target_shape)
        V4 = np.reshape(origin_image[INT_INDEX[0] + 1, INT_INDEX[1] + 1], target_shape)
        DIFF = INDEX - INT_INDEX
        H = np.tile(np.reshape(DIFF[0], target_size + (1,)), (1, 1, target_shape[2]))
        _H = 1 - H
        W = np.reshape(DIFF[1], target_size + (1,))
        W = np.tile(W, (1, 1, target_shape[2]))
        _W = 1 - W
        target_image = _W * _H * V1 + W * _H * V2 + _W * H * V3 + W * H * V4

        if target_image.shape[2] == 1:  # 返回二维数组型灰度图
            return target_image[:, :, 0]
        # 返回三通道彩色图
        return target_image


if __name__ == '__main__':
    img = imageio.imread("dog.jpg")
    p = Pyramid(img)
    # img = Pyramid.resize(img, (512, 512))
    print("高斯金字塔层数：", p.level + 1)
    print("原图尺寸：", img.shape)

    start = time.perf_counter()
    p.generate_pyramid()
    end = time.perf_counter()
    print("生成时间：", end - start)

    start = time.perf_counter()
    p.rebuild()
    end = time.perf_counter()
    print("重建时间：", end - start)

    gs, ls, bs = p.get_output()
    fig, axes = plt.subplots(3, len(gs), figsize=(16, 5.5))
    for i, pic in enumerate(gs):
        axes[0, i].set_title("gauss" + str(i))
        axes[0, i].imshow(pic)
    for j, pic in enumerate(ls):
        axes[1, j].set_title("laplace" + str(j))
        axes[1, j].imshow(pic)
    for k, pic in enumerate(bs):
        axes[2, k].set_title("build" + str(k))
        axes[2, k].imshow(pic)
    for i in range(3):
        for j in range(len(gs)):
            axes[i, j].axis("off")


    plt.show()
