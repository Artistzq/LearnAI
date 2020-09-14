import math
# import time

import imageio
import numpy as np
from matplotlib import pyplot as plt


class Rotate():
    @classmethod
    def rotate2D(cls, X, angle: float) -> np.ndarray:
        """
        该函数把一系列的坐标点 X ，转换为旋转后的坐标点，并返回

        Args:
            X (array-like): 一个 2xN 的矩阵，共有N个点，第一行存放的是每个点的x坐标，
            第二行存放的是y坐标
            angle (float): 逆时针旋转的度数，弧度制

        Returns:
            Xrot : 一个包含旋转后坐标的 2xN的 矩阵
        """
        X = np.asarray(X)
        assert X.ndim == 2 and X.shape[0] == 2, "传入2*N的矩阵"

        rotate_matrix = np.mat([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        ])
        Xrot = np.dot(rotate_matrix, X)
        return Xrot

    @classmethod
    def rotate(cls, I: np.ndarray, angle: float = 0, intepolate_grade=2) -> np.ndarray:
        """
        该函数返回一个图片 I 逆时针旋转 angle 角度后的图片，使用前向映射。

        Args:
            I (np.ndarray): 原图，以矩阵形式存储
            angle (float): 图片旋转的角度，逆时针，弧度制
            intepolate_grade (int, optional): 插值方法，默认2。

        Returns:
            np.ndarray: 旋转后的图片
        """
        # 预处理
        I = np.asarray(I)
        ndim = I.ndim
        assert ndim == 2 or ndim == 3
        if ndim == 2:
            I = np.expand_dims(I, axis=-1)
        height = I.shape[0]         # 原图行数
        width = I.shape[1]          # 原图列数
        N = np.size(I[:, :, 0])     # 原图像素点个数

        # 获取原图像素坐标序列，以图像左上角为原点，得到(2, N)的坐标序列
        meshgrid = np.meshgrid(range(height), range(width))
        coords_origin = np.resize(meshgrid, (2, N))
        # 旋转坐标
        coords_rot = cls.rotate2D(coords_origin, angle)
        coords_new = coords_rot - np.array([
            [np.min(coords_rot[0])],
            [np.min(coords_rot[1])]
        ])
        height_max, width_max = int(
            np.max(coords_new[0]))+1, int(np.max(coords_new[1]))+1
        # 多创建一行一列，防止越界
        target_img = np.zeros((height_max+1, width_max+1, I.shape[2]))

        # 对每个通道操作
        for num in range(I.shape[2]):
            target_channel = target_img[0:, 0:, num]
            origin_channel = I[:, :, num]

            if intepolate_grade == 0:
                # 直接取整
                coords_new = coords_new.astype(np.int32)  # 取整
                for i in range(N):
                    target_channel[coords_new[0, i], coords_new[1, i]] = \
                        origin_channel[coords_origin[0]
                                       [i], coords_origin[1, i]]
            if intepolate_grade == 1:
                # 考虑权重
                start = time.perf_counter()
                for i in range(N):
                    temp = np.zeros((height_max+1, width_max+1))
                    r, c = coords_new[0, i], coords_new[1, i]
                    a, b = r-int(r), c-int(c)
                    q = origin_channel[coords_origin[0]
                                       [i], coords_origin[1, i]]

                    target_channel[int(r), int(c)] += q*(1-a)*(1-b)
                    target_channel[int(r)+1, int(c)] += q*(1-b)*a
                    target_channel[int(r), int(c)+1] += q*(1-a)*b
                    target_channel[int(r)+1, int(c)+1] += q*a*b
                end = time.perf_counter()
                print(end - start)
            elif intepolate_grade == 2:
                # 考虑权重且归一化
                # 设置数组记录每个像素点处分到的权重和值
                target_weight = [
                    [[] for j in range(target_img.shape[1])] for i in range(target_img.shape[0])]
                for i in range(N):
                    # 双线性插值的逆操作，将四个点中心的值，按权重分配到四个点上
                    r, c = coords_new[0, i], coords_new[1, i]
                    a, b = r-int(r), c-int(c)
                    q = origin_channel[coords_origin[0]
                                       [i], coords_origin[1, i]]
                    target_weight[int(r)][int(c)].append([q, (1-a)*(1-b)])
                    target_weight[int(r)+1][int(c)].append([q, (1-b)*a])
                    target_weight[int(r)][int(c)+1].append([q, (1-a)*b])
                    target_weight[int(r)+1][int(c)+1].append([q, a*b])
                for i in range(target_img.shape[0]):
                    for j in range(target_img.shape[1]):
                        if len(target_weight[i][j]) != 0:
                            w_s = 0
                            for k in range(len(target_weight[i][j])):
                                w_s += target_weight[i][j][k][1]
                            for k in range(len(target_weight[i][j])):
                                target_channel[i][j] += target_weight[i][j][k][0] * \
                                    target_weight[i][j][k][1] / w_s

        target_img = (np.clip(target_img.astype(
            np.int32), 0, 255)).astype(np.uint8)
        if ndim == 2:
            return target_img[0:-1, 0:-1, 0]
        # 去掉最后一行、最后一列
        return target_img[0:-1, 0:-1]


if __name__ == "__main__":
    img = imageio.imread("crooked_horizon.jpg")
    angle = float(input("输入角度（弧度制）："))
    # img_rot_0 = Rotate.rotate(img, angle, intepolate_grade=0)
    # img_rot_1 = Rotate.rotate(img, angle, intepolate_grade=1)
    # start = time.perf_counter()
    img_rot_2 = Rotate.rotate(img, angle)
    # end = time.perf_counter()
    # print("耗时：", end-start, "秒")

    # imageio.imwrite("img_rot_0.jpg", img_rot_0)
    # imageio.imwrite("img_rot_1.jpg", img_rot_1)
    imageio.imwrite("img_rot_2.jpg", img_rot_2)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.subplot(1, 2, 2)
    plt.imshow(img_rot_2)
    plt.show()
