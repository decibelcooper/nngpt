import gc
import logging
import numpy as np
import random as rand
from scipy import spatial
import tensorflow as tf
import time


class Planar(object):
    def __init__(
        self,
        n_chans,
        chan_counts,
        height=100, width=100,
        m=50, n=50,
        sigma_l=1,
        sparse_cutoff=0.1,
        max_diff_sigma=2,
        sample_density=1000,
    ):
        self.n_chans = n_chans
        self.chan_counts = chan_counts
        self.height, self.width = height, width
        self.m, self.n = m, n
        self.sigma_l = sigma_l
        self.sparse_cutoff = sparse_cutoff
        self.max_diff_sigma = max_diff_sigma

        self._init_design(sample_density)
        self._init_tomo()

    def tomo(self, q, max_iter=100, ret_pixels=True):
        t0 = time.time()

        p = self.p
        retain = np.ones(len(self.Sigma_f_indices), dtype=np.bool)

        n_iter = 0

        while True:
            if len(p) == 0:
                feed_dict = {
                    self.Q: q,
                }
                s = np.reshape(self.S_init.eval(
                    feed_dict, session=self.sess), -1)
            else:
                feed_dict = {
                    self.P: p,
                    self.P_f_retain: retain,
                    self.Q: q,
                }
                s = np.reshape(self.S.eval(feed_dict, session=self.sess), -1)
            n_iter += 1

            if n_iter >= max_iter:
                logging.info('hit max iterations')
                break

            neg = np.flatnonzero(s < 0.01*np.max(s))
            if len(neg) == 0:
                break

            neg_pixels = p[neg]

            abandon_indices = np.concatenate(self.abandon_lookup[neg_pixels])
            np.put(retain, abandon_indices, False)

            p_new = np.setdiff1d(p, neg_pixels)
            if len(p_new) == 0:
                logging.info('p is null set')
                break
            else:
                p = p_new

        t1 = time.time()

        logging.info(f'tomography time: {t1-t0:.3f}')
        logging.info(f'tomography iterations: {n_iter}')
        logging.info(f'unconstrained pixel count: {len(p)}')

        if ret_pixels:
            f = np.zeros(self.g.shape[1])
            f[list(p)] = s
            return np.reshape(f, (self.m, self.n))

    def bin_channels(self, xy, **kwargs):
        t0 = time.time()

        q = np.zeros(self.n_chans)
        for chan, count in self.chan_counts(xy, **kwargs):
            if chan >= 0:
                q[chan] += count

        t1 = time.time()
        logging.info(f'binning time: {t1-t0:.3f}')

        return q

    def bin_pixels(self, xy, **kwargs):
        t0 = time.time()

        flat = np.zeros(self.m*self.n)
        for pixel, count in self.pixel_counts(xy, **kwargs):
            if pixel >= 0:
                flat[pixel] += count

        t1 = time.time()
        logging.info(f'binning time: {t1-t0:.3f}')

        return np.reshape(flat, (self.m, self.n))

    def get_mean_and_cov(self, img):
        x_pos = np.array(self.pixel_x)
        y_pos = np.array(self.pixel_y)
        x_pos2 = np.power(x_pos, 2)
        y_pos2 = np.power(y_pos, 2)
        xy_pos = x_pos * y_pos

        img_norm = np.reshape(img, -1) / np.sum(img)
        cross_mean = [x_pos2, y_pos2, xy_pos] @ img_norm
        p1 = [x_pos, y_pos, x_pos] @ img_norm
        p2 = [x_pos, y_pos, y_pos] @ img_norm
        mean = p1[0:2]
        cov = cross_mean - (p1) * (p2)
        cov = np.array([[cov[0], cov[2]], [cov[2], cov[1]]])

        return mean, cov

    def _init_design(self, sample_density):
        t0 = time.time()

        pitch_y = self.height / self.m
        pitch_x = self.width / self.n
        n_samples = int(sample_density * pitch_x * pitch_y)
        n_pixels = self.m * self.n

        # generate design matrix g
        self.g = np.zeros((self.n_chans, n_pixels))

        for pixel in range(0, n_pixels):
            i = pixel // self.n
            j = pixel % self.n
            pixel_y = self.height/2 - (i+0.5)*pitch_y
            pixel_x = self.width/2 - (j+0.5)*pitch_x

            xy = [pixel_x, pixel_y]
            xy += np.transpose([
                np.random.uniform(-pitch_x/2, pitch_x/2, n_samples),
                np.random.uniform(-pitch_y/2, pitch_y/2, n_samples),
            ])
            if self.max_diff_sigma > 0:
                xy += np.random.multivariate_normal(
                    [0.0, 0.0],
                    [[1.0, 0.0], [0.0, 1.0]],
                    len(xy),
                ) * np.reshape(
                    self.max_diff_sigma *
                    np.sqrt(np.random.uniform(0, 1, len(xy))),
                    (-1, 1),
                )

            for chan, count in self.chan_counts(xy, split=False, randomize=False):
                if chan >= 0:
                    self.g[chan][pixel] += count

        self.g /= 2*n_samples

        t1 = time.time()
        logging.info(f'g integration time: {t1-t0:.3f}')
        t0 = time.time()

        # generate sparse covariance matrix prior
        Sigma_f_indices = []
        Sigma_f_values = []
        abandon_lookup = [[] for i in range(0, n_pixels)]
        for i in range(0, n_pixels):
            inp = np.array([i for j in range(0, n_pixels)])
            jnp = np.array([j for j in range(0, n_pixels)])
            ri = (inp / self.n).astype(np.int64)
            ci = inp % self.n
            rj = (jnp / self.n).astype(np.int64)
            cj = jnp % self.n
            cov = np.exp(-(((ri-rj)*pitch_y)**2+((ci-cj)*pitch_x)
                           ** 2)/(2*self.sigma_l**2))

            indices = np.flatnonzero(cov > self.sparse_cutoff)
            values = cov[indices].astype(np.float32)
            j_indices = np.expand_dims(indices, axis=1)
            i_indices = np.expand_dims(
                [i for j in range(0, len(indices))], axis=1)
            ij_indices = np.concatenate((i_indices, j_indices), axis=1)

            abandon_lookup[i].extend(
                [i for i in range(len(Sigma_f_values), len(
                    Sigma_f_values) + len(values))],
            )
            for k in range(0, len(indices)):
                j = indices[k]
                if j == i:
                    continue
                abandon_lookup[j].append(len(Sigma_f_values) + k)

            Sigma_f_indices.extend(ij_indices)
            Sigma_f_values.extend(values)

        Sigma_f_indices = np.array(Sigma_f_indices)
        Sigma_f_values = np.array(Sigma_f_values)
        abandon_lookup = np.array(abandon_lookup)
        self.Sigma_f_indices = Sigma_f_indices
        self.Sigma_f_values = Sigma_f_values
        self.abandon_lookup = abandon_lookup

        t1 = time.time()
        logging.info(f'covariance matrix calculation time: {t1-t0:.3f}')

        n_pixels = self.m * self.n
        self.pixel_y = np.array([self.height/2 - (i//self.n+0.5) * pitch_y
                                 for i in range(0, n_pixels)])
        self.pixel_x = np.array([self.width/2 - (i % self.n+0.5) * pitch_x
                                 for i in range(0, n_pixels)])

        # fast nearest pixel calculation using native KDTree implementation
        pixel_tree = spatial.cKDTree(
            np.transpose([self.pixel_x, self.pixel_y]))

        def pixel_counts(xy):
            _, pixels = pixel_tree.query(xy)
            pixels, counts = np.unique(pixels, return_counts=True)
            return zip(pixels, counts)
        self.pixel_counts = pixel_counts

    def _init_tomo(self):
        t0 = time.time()

        self.p = np.array([i for i in range(0, self.g.shape[1])])

        # Setup tensorflow graph
        gc.collect()
        graph = tf.Graph()
        graph.as_default()
        self.sess = tf.compat.v1.Session()

        self.P = tf.compat.v1.placeholder(dtype=tf.int32, shape=(None,))
        self.P_f_retain = tf.compat.v1.placeholder(dtype=tf.bool)
        self.Q = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))

        Q = tf.reshape(self.Q, (-1, 1))
        q_sum = tf.reduce_sum(Q)
        Sigma = tf.compat.v1.matrix_diag(tf.abs(self.Q) + 1)
        G = tf.convert_to_tensor(self.g, dtype=tf.float32)
        G_T = tf.transpose(G)
        G_P_T = tf.gather(G_T, self.P, axis=0)
        G_P = tf.transpose(G_P_T)
        P_f = tf.sparse.SparseTensor(
            indices=self.Sigma_f_indices,
            values=4 * q_sum * self.Sigma_f_values,
            dense_shape=(self.g.shape[1], self.g.shape[1]),
        )
        P_f_P = tf.sparse.retain(P_f, self.P_f_retain)
        A = tf.sparse.sparse_dense_matmul(P_f, G_T)
        A_P = tf.gather(tf.sparse.sparse_dense_matmul(P_f_P, G_T), self.P)

        L = tf.linalg.cholesky(Sigma + G @ A)
        Q_P = tf.linalg.cholesky_solve(L, Q)
        self.S_init = A @ Q_P

        L = tf.linalg.cholesky(Sigma + G_P @ A_P)
        Q_P = tf.linalg.cholesky_solve(L, Q)
        self.S = A_P @ Q_P

        t1 = time.time()
        logging.info(f'setup time: {t1-t0:.3f}')
