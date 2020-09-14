import numpy as np
import cv2  # 3.4.1.5
import matplotlib.pyplot as plt
import math
import copy


class SIFT:
    CONTR_T = 0.04      # 对比度阈值
    MAX_ITER_STEPS = 5  # 插值寻找极值点迭代次数
    GAMMA = 10          # 边缘响应阈值
    SCALE = 255         # 插值寻找极值点时用到的比例
    BORDER = 1          # 图像边缘

    def __init__(self, image, octave: int = -1, S: int = 3, sigma: float = 1.52, kernel_size=15):
        """
        初始化金字塔
        :param image: 原图
        :param octave: 组数，默认最大值
        :param S: 每层图片需要检测的尺度，3-5，默认为3
        :param sigma: 高斯滤波器参数
        """
        self.image: np.ndarray = np.array(image)
        if self.image.ndim == 3:
            self.image = self.image[:, :, 0]
        self.S = S
        k = math.pow(2, 1 / S)

        self.sigmas = [sigma]  # [1.52] sqrt(1.6^2 - 1^2)
        for i in range(1, S + 3):
            sig_pre = math.pow(k, i - 1) * sigma
            sig_tot = sig_pre * k
            self.sigmas.append(math.sqrt(sig_tot ** 2 - sig_pre ** 2))
        self.kernel_size = kernel_size

        self.max_octave = math.ceil(np.log2(min(image.shape[:2]))) + 1
        assert self.max_octave > 3, "图像太小"
        if octave < 0 or octave > self.max_octave - 3:
            octave = self.max_octave - 3
        self.octave = octave

        self.gauss_pyr = []
        self.dog_pyr = []
        self.key_points = []
        self.descriptors = []
        self.locations = []

    def get_gauss_pyr(self):
        return copy.deepcopy(self.gauss_pyr)

    def get_dog_pyr(self):
        return copy.deepcopy(self.dog_pyr)

    def build_gauss_pyr(self):
        """生成高斯金字塔"""
        self.gauss_pyr = [[] for o in range(self.octave)]
        for o in range(self.octave):
            for i in range(self.S + 3):
                if o == 0 and i == 0:
                    # -1层（0组），不扩大，使用原图像作为基层
                    base = self.image
                    self.gauss_pyr[o].append(base)
                elif o != 0 and i == 0:
                    # 本组的第一层由上一组的倒数第3层降采样，上一组已经经过滤波，所以直接降采样
                    down = self.down_sample(self.gauss_pyr[o - 1][-3])
                    self.gauss_pyr[o].append(down)
                else:
                    # 防止卷积核尺寸比半径大
                    min_radius = min(self.gauss_pyr[o][0].shape + (self.kernel_size,))
                    img = self.gaussian_filter(self.gauss_pyr[o][i-1], self.sigmas[i], min_radius)
                    self.gauss_pyr[o].append(img)
                # print("第", o, "组第", i + 1, "层建完了", self.gauss_pyr[o][i].shape)
            # print("第", o, "组建完了")
        return self.gauss_pyr

    def build_dog_pyr(self):
        """生成高斯差分金字塔"""
        assert self.gauss_pyr is not None, "先建造高斯金字塔"
        self.dog_pyr = [[self.gauss_pyr[o][i + 1] - self.gauss_pyr[o][i]
                         for i in range(self.S + 2)]
                        for o in range(self.octave)]
        return self.dog_pyr

    def find_features(self):
        """寻找特征点"""
        self.build_gauss_pyr()
        self.build_dog_pyr()
        threshold = int(0.5 * self.CONTR_T / self.S * self.SCALE)
        self.key_points.clear()
        self.locations.clear()
        for o in range(self.octave):
            dog_o = np.asarray(self.dog_pyr[o])
            if dog_o.shape[1] < 2 * self.BORDER or dog_o.shape[2] < 2 * self.BORDER:
                break
            pixels = np.zeros(dog_o.shape + (3, 3, 3))
            for i in range(1, dog_o.shape[0] - 1):
                for r in range(self.BORDER, dog_o.shape[1] - self.BORDER):
                    for c in range(self.BORDER, dog_o.shape[2] - self.BORDER):
                        pixels[i, r, c] = dog_o[i - 1: i + 2, r - 1:r + 2, c - 1:c + 2]
            maxs = np.max(np.fabs(pixels), axis=(-1, -2, -3))
            vals = np.fabs(dog_o)
            # 大于阈值，且是极值（邻域内最大/小值就是自己）
            idx = np.where(np.where((maxs == vals) & (vals > threshold), True, False))
            idx = np.array(idx).T  # (i, r, c)
            for i, r, c in idx:
                save_loc = np.array([o, i, r, c])
                kpt = cv2.KeyPoint()
                # 插值寻找极值点的精确位置
                if not self.adjust_extrema(self.dog_pyr, (o, i, r, c), kpt, save_loc):
                    continue
                # 计算幅角（直方图统计）
                self.cal_orient_hist(self.gauss_pyr, kpt, save_loc, self.key_points, self.locations)

        # 计算特征向量
        self.cal_descriptors(self.gauss_pyr, self.key_points, self.locations, self.descriptors)
        self.descriptors = np.array(self.descriptors, dtype=np.float32)
        self.descriptors = np.around(self.descriptors)
        return self.key_points, self.descriptors

    def adjust_extrema(self, pyr, location, kpt, save_loc):
        """
        插值，在最值点的基础上寻找真正的极值点
        :param pyr: 金字塔（DOG）
        :param location: 最值点坐标 (o组, i层, r高(行), c宽（列）)
        :param kpt: 特征点，存储真正极值点的部分信息
        :param save_loc: 真正极值点所在层数、组数以及坐标
        :return: 是否是精确极值点
        """
        (o, i, r, c) = location
        o_size = pyr[o][0].shape
        img_scale = 1 / self.SCALE
        Xd = np.zeros((3, 1))
        for _ in range(self.MAX_ITER_STEPS):
            # 迭代
            dD = img_scale * 0.5 * np.array([[pyr[o][i][r][c + 1] - pyr[o][i][r][c - 1],
                                              pyr[o][i][r + 1][c] - pyr[o][i][r - 1][c],
                                              pyr[o][i + 1][r][c] - pyr[o][i - 1][r][c]]]).T
            f0 = pyr[o][i][r][c]
            dxx = img_scale * (pyr[o][i][r][c + 1] + pyr[o][i][r][c - 1] - 2 * f0)
            dyy = img_scale * (pyr[o][i][r + 1][c] + pyr[o][i][r - 1][c] - 2 * f0)
            dss = img_scale * (pyr[o][i + 1][r][c] + pyr[o][i - 1][r][c] - 2 * f0)
            dxy = img_scale * 0.25 * (
                    pyr[o][i][r + 1][c + 1] + pyr[o][i][r - 1][c - 1] - pyr[o][i][r - 1][c + 1] - pyr[o][i][r + 1][
                c - 1])
            dxs = img_scale * 0.25 * (pyr[o][i + 1][r + 1][c] + pyr[o][i - 1][r - 1][c] - pyr[o][i + 1][r - 1][c] -
                                      pyr[o][i - 1][r + 1][c])
            dys = img_scale * 0.25 * (
                    pyr[o][i + 1][r][c + 1] + pyr[o][i - 1][r][c - 1] - pyr[o][i - 1][r][c + 1] - pyr[o][i + 1][r][
                c - 1])
            dDD = np.reshape(np.array([dxx, dxy, dxs, dxy, dyy, dys, dxs, dys, dss]), (3, 3))
            # if np.linalg.det(dDD) == 0.0: return False
            Xd = np.reshape(np.linalg.solve(dDD, -dD), (3,))  # 偏离量，shape：(3, ), (c, r, i)
            # 新坐标
            c, r, i = int(round(c + Xd[0])), int(round(r + Xd[1])), int(round(i + Xd[2]))

            # 坐标偏离量小于0.5，足够精确，退出迭代
            if (np.fabs(Xd) < 0.5).all():
                break
            # 太大，不是特征点
            if (np.abs(Xd) > 65535).any():
                return False
            # 超出图像范围，不是特征点
            if i < 1 or i > self.S or r < self.BORDER or r >= o_size[0] - self.BORDER - 1 or \
                    c < self.BORDER or c >= o_size[1] - self.BORDER - 1:
                return False

        # 响应值是否稳定
        dD = img_scale * 0.5 * np.array([[pyr[o][i][r][c + 1] - pyr[o][i][r][c - 1],
                                          pyr[o][i][r + 1][c] - pyr[o][i][r - 1][c],
                                          pyr[o][i + 1][r][c] - pyr[o][i - 1][r][c]]]).transpose()
        t = np.dot(np.reshape(dD, (3,)), np.reshape(Xd, (3,)))
        contr = pyr[o][i][r][c] * img_scale + t * 0.5
        if abs(contr) * self.S < self.CONTR_T:
            return False

        # 消除边缘响应
        f0 = pyr[o][i][r][c]
        dxx = img_scale * (pyr[o][i][r][c + 1] + pyr[o][i][r][c - 1] - 2 * f0)
        dyy = img_scale * (pyr[o][i][r + 1][c] + pyr[o][i][r - 1][c] - 2 * f0)
        dxy = img_scale * 0.25 * (
                pyr[o][i][r + 1][c + 1] + pyr[o][i][r - 1][c - 1] - pyr[o][i][r - 1][c + 1] - pyr[o][i][r + 1][
            c - 1])
        tr = dxx + dyy
        det = dxx * dyy - dxy ** 2
        if det <= 0 or self.GAMMA * (tr ** 2) >= det * ((1 + self.GAMMA) ** 2):
            return False

        Xd = np.around(Xd).astype(np.int32)
        # 跳出循环前，o,i,r,c已更新
        save_loc[:] = np.array([o, i, r, c])
        kpt.response = abs(contr)
        kpt.octave = int(save_loc[0] + save_loc[1] * (2 ** 8) + round((Xd[2] + 0.5) * 255) * (2 ** 16))
        kpt.pt = (save_loc[3]) * math.pow(2, save_loc[0]), save_loc[2] * math.pow(2, save_loc[0])  # 相对于第0组的坐标
        kpt.size = self.sigmas[0] * math.pow(2, save_loc[0]) * math.pow(2, save_loc[1] / self.S) * 2

        return True

    def cal_orient_hist(self, pyr, kpt, save_loc, key_points, saved_locs):
        """
        计算直方图，计算出特征点的方向，并加入到特征值列表中
        :param pyr: 金字塔（高斯尺度空间）
        :param kpt: 特征点
        :param save_loc: 当前组、层，像素坐标
        :param key_points: 特征点列表
        :param saved_locs: 把特征点组、层保存到saved_locs中
        :return: None
        """
        o, i, r, c = save_loc
        o_size = pyr[o][0].shape
        # 由kpt.size = sigma[0] * 2**o * 2**(i/S) * 2
        # 可得，sigma = sigma[0] * 2**(i/S)，即相对本组基准层
        sigma = kpt.size / (2 * (2 ** o)) * 1.5
        radius = int(round(3 * sigma))
        lf, ri, up, do = c - radius, c + radius, r - radius, r + radius
        L = pyr[o][i]

        shape = (do - up + 1, ri - lf + 1)

        # 代替循环
        cx_ry = np.meshgrid(range(c - radius, c + radius + 1), range(r - radius, r + radius + 1), indexing="ij")
        cx_ry = np.array(cx_ry).reshape((2, (2 * radius + 1) ** 2))
        meet_cr = np.where((cx_ry[0] >= self.BORDER) & (cx_ry[0] < o_size[1] - self.BORDER - 1) &
                           (cx_ry[1] >= self.BORDER) & (cx_ry[1] < o_size[0] - self.BORDER - 1),
                           True, False)  # (满足条件的坐标个数, )
        meet_cr = np.reshape(meet_cr, (meet_cr.size,))
        cx_ry = cx_ry[:, meet_cr]  # (2, 满足条件) 满足条件的坐标矩阵
        rela_cx_ry = cx_ry - np.array([[c], [r]])  # (2, 满足条件) # 相对坐标，用于计算高斯加权矩阵
        dx = L[cx_ry[1], [cx_ry[0] + 1]] - L[cx_ry[1], cx_ry[0] - 1]
        dy = L[cx_ry[1] + 1, [cx_ry[0]]] - L[cx_ry[1] - 1, cx_ry[0]]
        mag = np.linalg.norm([dx, dy], axis=0).flatten()
        theta = np.arctan2(dy, dx).flatten()  # -pi ~ pi
        theta[np.where(theta < 0)] += 2 * math.pi  # 0~2pi
        weight = self.gaussian_kernel(sigma, 2 * radius + 1)
        weight = weight[rela_cx_ry[0], rela_cx_ry[1]]
        mag = mag * weight

        # 直方图36根柱子, 方向从0-10度开始，10-20,20-30...
        cols = 36
        hist = np.zeros((cols,))
        deg_step = 360 / cols
        starts = math.pi / 180 * np.arange(0, 360, deg_step)
        ends = math.pi / 180 * np.arange(0 + deg_step, 360 + deg_step, deg_step)
        for col in range(cols):
            indices = np.where((ends[col] > theta) & (theta >= starts[col]))
            hist[col] += np.sum(mag[indices])

        # 平滑直方图，两次
        # 圆周循环
        idx = np.arange(1, cols + 1)
        idx[-1] = 0
        for _ in range(2):
            hist = 0.25 * hist[np.arange(-1, cols - 1)] + 0.5 * hist[np.arange(cols)] + 0.25 * hist[idx]

        # 寻找辅方向，与主方向一起加入方向列表
        main_orient = np.argmax(hist)
        vice_orients = list(np.where((hist >= 0.8 * np.max(hist)) & (hist != np.max(hist)))[0])
        orients = [main_orient] + vice_orients

        # 为主方向、次方向拟合精确的方向角度
        # 若辅方向不满足条件，抛出
        # 圆周循环
        angles = []
        for ori in orients:
            pre, cur = hist[ori - 1], hist[ori]
            if ori == cols - 1:
                nex = hist[0]
            else:
                nex = hist[ori + 1]
            if cur < pre or cur < nex:
                orients.remove(ori)
                continue
            bin = ori + (pre - nex) / (2 * (pre + nex - 2 * cur))
            bin = (bin + cols) if bin < 0 else (bin - cols) if bin >= cols else bin
            angle = 360 - bin * 360 / cols
            if abs(angle - 360) < 0.1:
                angle = 0
            angles.append(angle)

        # 主、辅方向加入特征点列表
        for angle in angles:
            temp = (kpt.pt, kpt.size, kpt.response, kpt.octave, kpt.class_id)
            n_kpt = cv2.KeyPoint(x=temp[0][0], y=temp[0][1], _size=temp[1], _response=temp[2],
                                 _octave=temp[3], _class_id=temp[4])
            n_kpt.angle = angle
            key_points.append(n_kpt)
            saved_locs.append(save_loc)

    def cal_descriptors(self, pyr, key_points, locations, descriptors):
        """
        计算特征点列表中列表的特征描述
        :param pyr: 金字塔，此处为高斯金字塔
        :param key_points: 关键点列表
        :param locations: 关键点所在层数、组数列表（i, o）
        :param descriptors: 描述向量
        :return: 描述向量
        """
        # 遍历所有特征点
        # print("计算描述子")
        n = descr_hist_bins = 8
        d = descr_width = 4
        for seq, (kpt, loc) in enumerate(zip(key_points, locations)):
            o, i = loc[:2]  # 组，层
            scale = math.pow(2, -o)
            r, c = kpt.pt[1] * scale, kpt.pt[0] * scale  # 列坐标，行坐标
            (o_size_r, o_size_c) = pyr[o][0].shape[:2]
            sigma = kpt.size * scale
            tri_sigma = 3 * sigma  # 3σ
            y, x = int(r * scale), int(c * scale)  # 第o组图像的坐标
            angle = (360 - kpt.angle) * math.pi / 180  # 特征点的角度(0-2pi)
            image = pyr[o][i]  # 第o组第i层图像
            radius = round(3 * sigma * math.sqrt(2) * (d + 1) / 2)  # 邻域半径
            radius = min(o_size_r, o_size_c, radius)
            exp_scale = -1 / ((d ** 2) / 2)  # 高斯加权

            cos_t, sin_t = math.cos(angle), math.sin(angle)
            rot_mat = np.array([
                [cos_t, -sin_t],
                [sin_t, cos_t]
            ]) / tri_sigma  # 除以3sigma（小方格边长），使参考系为小方格，每个小方格坐标+1
            # 生成直方图框架
            hist = np.zeros((d + 2, d + 2, n + 2))  # 留出2个位置处理插值越界

            # 用矩阵操作完成循环，找到符合要求的特征点的坐标
            i_j = np.meshgrid(range(-radius, radius + 1), range(-radius, radius + 1), indexing="ij")
            i_j = np.array(i_j).reshape((2, (2 * radius + 1) ** 2))
            # 旋转矩阵已经除以3σ，此处不需再除
            r_c_rot = np.dot(rot_mat, i_j)  # (2, len(kpts))
            r_c_bin = r_c_rot + d / 2 - 0.5  # (2, len*kpts))
            r_c_neigh = np.add(i_j, np.array([[y], [x]]))  # (2, len(kpts))

            meet_idx = np.where((r_c_bin[0] < d) & (r_c_bin[0] >= 0) & (r_c_bin[1] < d) & (r_c_bin[1] >= 0) &
                                (r_c_neigh[0] < o_size_r - 1) & (r_c_neigh[0] > 0) & (r_c_neigh[1] < o_size_c - 1) &
                                (r_c_neigh[1] > 0), True, False)  # (满足条件的坐标个数, )
            meet_idx = np.reshape(meet_idx, (meet_idx.size,))
            r_c_rot = r_c_rot[:, meet_idx]  # (2, len)
            r_c_bin = r_c_bin[:, meet_idx]  # (2, len)
            r_c_neigh = r_c_neigh[:, meet_idx]  # (2, len)
            dx = image[r_c_neigh[0], r_c_neigh[1] + 1] - image[r_c_neigh[0], r_c_neigh[1] - 1]  # (len,)
            dy = image[r_c_neigh[0] - 1, r_c_neigh[1]] - image[r_c_neigh[0] + 1, r_c_neigh[1]]  # (len, )

            # 计算对应的梯度、幅值和权重
            mag = np.linalg.norm([dx, dy], axis=0)
            theta = np.arctan2(dy, dx)  # (-pi-pi)
            theta[np.where(theta < 0)] += 2 * math.pi  # 0~2pi
            # （x^2 + y^2）/(2σ^2)
            weights = np.power(np.linalg.norm(r_c_rot, axis=0), 2)
            weights = np.exp(weights * exp_scale)
            mag = mag * weights

            # 计算整数位置和小数偏移，为三线性插值做准备
            # 三位直方图的高
            ori_bins = (theta - angle) * n / (2 * math.pi)
            # 三维坐标的整数部分
            r0 = np.floor(r_c_bin[0]).astype(np.int)
            c0 = np.floor(r_c_bin[1]).astype(np.int)
            o0 = np.floor(ori_bins).astype(np.int)
            # 三维坐标的小数部分
            rf, cf, of = r_c_bin[0] - r0, r_c_bin[1] - c0, ori_bins - o0
            # 圆周循环
            o_low_0 = np.where(o0 < 0)
            o_above_360 = np.where(o0 >= n)
            o0[o_low_0] += n
            o0[o_above_360] -= n

            # 三线性插值
            v_r1 = mag * rf
            v_r0 = mag - v_r1
            v_rc11, v_rc01 = v_r1 * cf, v_r0 * cf
            v_rc10, v_rc00 = v_r1 - v_rc11, v_r0 - v_rc01
            v_rco111, v_rco101, v_rco011, v_rco001 = v_rc11 * of, v_rc10 * of, v_rc01 * of, v_rc00 * of
            v_rco110, v_rco100, v_rco010, v_rco000 = v_rc11 - v_rco111, v_rc10 - v_rco101, v_rc01 - v_rco011, v_rc00 - v_rco001

            # 直方图统计
            for vv, rv, cv, ov in zip(v_rco000, r0, c0, o0):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco001, r0, c0, o0 + 1):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco010, r0, c0 + 1, o0):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco011, r0, c0 + 1, o0 + 1):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco100, r0 + 1, c0, o0):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco101, r0 + 1, c0, o0 + 1):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco110, r0 + 1, c0 + 1, o0):
                hist[rv, cv, ov] += vv
            for vv, rv, cv, ov in zip(v_rco111, r0 + 1, c0 + 1, o0 + 1):
                hist[rv, cv, ov] += vv

            # 圆周循环，展开特征向量
            hist[:, :, 0] += hist[:, :, n]
            hist[:, :, 1] += hist[:, :, n + 1]
            hist = hist[:d, :d, :n]
            dst = hist.flatten()

            # 特征向量最后的处理
            norm2 = np.linalg.norm(dst)
            thr = norm2 * 0.2
            dst[np.where(dst >= thr)] = thr
            norm2 = np.linalg.norm(dst)
            norm2 = 512. / max(norm2, 0.01)
            dst = dst * norm2

            descriptors.append(dst)

        return descriptors

    def up_sample(self, image, sigma=None, scale=2):
        """
        升采样
        :param image: 图像
        :param sigma: 高斯平滑参数，若为空，则不平滑
        :param scale: 比例，默认位2，即放大两倍
        :return: 升采样的图像
        """
        if scale == 1:
            return image
        oh, ow = image.shape[0], image.shape[1]
        up = self.resize(image, (scale * oh, scale * ow))
        if sigma is not None:
            up = self.gaussian_filter(up, sigma, self.kernel_size)
        return up

    def down_sample(self, image, sigma=None, scale=2):
        """
        降采样
        :param image: 图像
        :param sigma: 高斯平滑参数，若为None，则不进行高斯平滑
        :param scale: 比例，默认2，即缩小2倍
        :return: 降采样的图像
        """
        oh, ow = image.shape[0], image.shape[1]
        min_radius = min(oh, ow, self.kernel_size)
        origin_image = image
        if sigma is not None:
            origin_image = self.gaussian_filter(image, sigma, min_radius)
        down = self.resize(origin_image, ((oh + 1) // scale, (ow + 1) // scale))
        return down

    @classmethod
    def gaussian_filter(cls, image, sigma, k_size):
        o_shape = image.shape
        border = k_size - (k_size + 1) // 2
        # 填充图像，防止滤波后图像缩小
        image = np.pad(image,
                       ((border, border), (border, border)), mode="symmetric")
        kernel = cls.gaussian_kernel(sigma, k_size, is2D=False)
        # 先进行图像宽方向上的卷积，对于一维卷积，不必遍历像素，直接遍历卷积核，
        # 把卷积核乘到图像整体矩阵中，再将得到的乘积矩阵对应位相加，就等价于遍历图像
        gauss_w = np.zeros((image.shape[0], o_shape[1]))
        for i, v in enumerate(kernel):
            gauss_w += v * image[:, i: o_shape[1] + i]
        # 再在完成宽方向卷积的基础上，进行图像高方向上的卷积
        gauss_h = np.zeros(o_shape[:2])
        for i, v in enumerate(kernel):
            gauss_h += v * gauss_w[i: o_shape[0] + i, :]
        return gauss_h

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
                    l = (l - a) * 255. / (b - a)
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

    @classmethod
    def gaussian_kernel(cls, sigma, size, is2D=True):
        """返回高斯核"""
        # 不乘高斯函数前面的系数，因为需要归一化，所以节省时间
        if is2D:
            def func(x, y):
                return math.e ** ((-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2))

            kernel = np.fromfunction(func, (size, size))
        else:
            def func(x):
                return math.e ** ((-1 * (x - (size - 1) / 2) ** 2) / (2 * sigma ** 2))

            kernel = np.fromfunction(func, (size,))
        return kernel / np.sum(kernel)

class ShowAlg:
    """个人算法演示类"""

    @classmethod
    def show(cls, image):
        s = SIFT(image)
        kpts, features = s.find_features()
        gauss_pyr = s.get_gauss_pyr()
        dog_pyr = s.get_dog_pyr()
        cls.show_kpts(image, kpts)
        cls.show_pyr(gauss_pyr)
        cls.show_pyr(dog_pyr)
        pass

    @classmethod
    def show_kpts(cls, image, key_points):
        plt.figure(figsize=(10, 6))
        img_dis = image.copy()
        scale = 1

        start_points = np.array([(pt.pt[0], pt.pt[1]) for pt in key_points])
        start_points = start_points * scale

        angles = np.array([pt.angle * math.pi / 180 for pt in key_points])
        dev = 1 * np.array(
            [(pt.size * math.cos(angle), pt.size * math.sin(angle)) for pt, angle in zip(key_points, angles)])
        show = dev[np.where(np.isnan(dev))]
        end_points = start_points + dev

        start_points = np.around(start_points).astype(np.int32)
        end_points = np.around(end_points).astype(np.int32)
        for i in range(len(start_points)):
            sx, sy = start_points[i][0], start_points[i][1]
            ex, ey = end_points[i][0], end_points[i][1]
            cv2.circle(img_dis, (sx, sy), 1, (255, 255, 255), 2)
            cv2.arrowedLine(img_dis, (sx, sy), (ex, ey), (255,255,255), 1)
        plt.title("keypoints and angles")
        plt.imshow(img_dis)

    @classmethod
    def show_pyr(self, pyramid):
        fig, axes = plt.subplots(len(pyramid), len(pyramid[0]), figsize=(16, 7))
        for i, pics in enumerate(pyramid):
            for j, pic in enumerate(pics):
                pic = SIFT.format(pic)
                axes[i, j].imshow(pic)
                axes[i, j].set_title(str(pic.shape))
                axes[i, j].axis("off")
        fig.tight_layout()


class Stitcher:
    """图像拼接类"""

    @classmethod
    def stitch(cls, images, ratio=0.75, reproj_thresh=4.0, use_my_code=True, show_matches=True, show_step=3):
        """
        图像拼接
        :param images: 图像拼接列表，(image1, image2)
        :param ratio: 最好配对的比例
        :param reproj_thresh: RANSAC算法阈值
        :param use_my_code: 是否使用自己的代码
        :return: 拼接后的图片
        """
        # 读取并检测
        (imageB, imageA) = images
        (kptsA, featuresA) = cls.detect_and_compute(imageA, use_my_code)
        (kptsB, featuresB) = cls.detect_and_compute(imageB, use_my_code)

        # 特征匹配
        M = cls.match_keypoints(kptsA, kptsB, featuresA, featuresB, ratio, reproj_thresh)
        if M is None: return None

        # 获取单应矩阵，并进行透视变换
        (good, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))

        # 实现图像覆盖拼接
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        match_img = None
        if show_matches:
            match_img = cv2.drawMatchesKnn(imageA, kptsA, imageB, kptsB, good[::show_step], None, flags=2)
        return result, match_img

    @classmethod
    def detect_and_compute(cls, image, use_my_code):
        """
        检测关键点，生成特征向量
        :param image:
        :param use_my_code:
        :return: 特征点，特征向量
        """
        # 转换为单通道
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not use_my_code:
            # opencv方法
            describe = cv2.xfeatures2d.SIFT_create()
            (kps, features) = describe.detectAndCompute(gray, None)
        else:
            # 自己的方法
            s = SIFT(gray)
            (kps, features) = s.find_features()
        return kps, features

    @classmethod
    def match_keypoints(cls, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        """
        配对
        :param kpsA: 特征点列表A
        :param kpsB: 特征点列表B
        :param featuresA: 特征向量集A
        :param featuresB: 特征向量集B
        :param ratio: 选择较好配对的比例
        :param reprojThresh: ransac算法阈值
        :return: 匹配，单应矩阵，状态
        """
        # 暴力匹配
        matches = cv2.BFMatcher()
        # 配对
        rawMatches = matches.knnMatch(featuresA, featuresB, 2)
        # 选出较好的配对
        matches = []  # 坐标
        good = []  # 配对结果，返回的匹配结果
        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                good.append([m[0]])
        # 当大于4个配对点就可以求得单应矩阵
        if len(matches) > 4:
            # 获取配对点的坐标
            kpsA = np.float32([kp.pt for kp in kpsA])
            kpsB = np.float32([kp.pt for kp in kpsB])
            ptsA = np.float32([kpsA[i] for _, i in matches])
            ptsB = np.float32([kpsB[i] for i, _ in matches])
            # 求单应矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return good, H, status

if __name__ == '__main__':
    image1 = cv2.imread("left_480p.jpg")
    image2 = cv2.imread("right_480p.jpg")

    # 原图
    fig, axes1 = plt.subplots(1, 2, figsize=(10, 5))
    axes1[0].imshow(image1)
    axes1[0].set_title("left")
    axes1[1].imshow(image2)
    axes1[1].set_title("right")

    print("个人算法演示(以图象一为例，SIFT特征提取)...")
    ShowAlg.show(image1)
    print("结束")

    print("OpenCV 拼接中...")
    result_cv, match_cv = Stitcher.stitch((image1, image2), use_my_code=False, show_matches=True)
    print("OpenCV 拼接结束")

    print("个人代码 拼接中...")
    result_my, match_my = Stitcher.stitch((image1, image2), use_my_code=True, show_matches=True)
    print("个人代码 拼接结束")

    # 显示拼接结果
    stitch_result_fig, axes2 = plt.subplots(2, 1, figsize=(14, 8))
    axes2[0].imshow(result_cv)
    axes2[0].set_title("OpenCV stitch")
    axes2[1].imshow(result_my)
    axes2[1].set_title("my sift stitch")

    # 显示匹配结果
    if match_cv is not None:
        match_fig_cv, axes3 = plt.subplots(1, 1, figsize=(10, 5))
        axes3.set_title("OpenCV matches")
        axes3.imshow(match_cv)

    if match_my is not None:
        match_fig_my, axes4 = plt.subplots(1, 1, figsize=(10, 5))
        axes4.set_title("mycode matches")
        axes4.imshow(match_my)

    plt.show()