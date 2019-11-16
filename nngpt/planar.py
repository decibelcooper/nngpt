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

    def tomo(self, d, ret_pixels=True):
        t0 = time.time()

        iters, phi_p, p = self.sess.run(
            (self.iters, self.phi_p, self.p), feed_dict={self.d: d})

        t1 = time.time()

        logging.info(f'tomography time: {t1-t0:.3f}')
        logging.info(f'tomography iterations: {iters}')
        logging.info(f'unconstrained pixel count: {len(p)}')

        if ret_pixels:
            phi = np.zeros(self.G.shape[1])
            phi[p] = phi_p
            return np.reshape(phi, (self.m, self.n))

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
        self.G = np.zeros((self.n_chans, n_pixels))

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
                    self.G[chan][pixel] += count

        self.G /= 2*n_samples

        t1 = time.time()
        logging.info(f'g integration time: {t1-t0:.3f}')
        t0 = time.time()

        # generate sparse covariance matrix prior
        Sigma_p_indices = []
        Sigma_p_values = []
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
                [i for i in range(len(Sigma_p_values), len(
                    Sigma_p_values) + len(values))],
            )
            for k in range(0, len(indices)):
                j = indices[k]
                if j == i:
                    continue
                abandon_lookup[j].append(len(Sigma_p_values) + k)

            Sigma_p_indices.extend(ij_indices)
            Sigma_p_values.extend(values)

        Sigma_p_indices = np.array(Sigma_p_indices)
        Sigma_p_values = np.array(Sigma_p_values)
        abandon_lookup = np.array(abandon_lookup)
        self.Sigma_prior_indices = Sigma_p_indices
        self.Sigma_prior_values = Sigma_p_values
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
        # Setup tensorflow graph
        graph = tf.Graph()
        graph.as_default()
        self.sess = tf.compat.v1.Session()
        gc.collect()

        # inputs
        self.d = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None,))
        G = tf.convert_to_tensor(self.G, dtype=tf.float32)
        Sigma_prior = tf.sparse.SparseTensor(
            indices=self.Sigma_prior_indices,
            values=4 * tf.reduce_sum(self.d) * self.Sigma_prior_values,
            dense_shape=(self.G.shape[1], self.G.shape[1]),
        )
        abandon_lookup = tf.ragged.constant(self.abandon_lookup)

        # prepared inputs
        d = tf.reshape(self.d, (-1, 1))
        Sigma_d = tf.compat.v1.matrix_diag(tf.abs(self.d) + 1)
        G_T = tf.transpose(G)

        # define step loop body for iterations that require an update
        def update(index, phi_p, p, Sigma_prior_retain):
            # find solution to phi given the particular passive set p
            G_p = tf.gather(G, p, axis=1)
            Sigma_prior_p = tf.sparse.retain(
                Sigma_prior, Sigma_prior_retain)
            A_p = tf.gather(tf.sparse.sparse_dense_matmul(
                Sigma_prior_p, G_T), p)
            decomp = tf.linalg.cholesky(Sigma_d + G_p @ A_p)
            d_w = tf.linalg.cholesky_solve(decomp, d)
            phi_p = tf.reshape(A_p @ d_w, (-1,))

            # determine pixel indices that are negative, and remove them from p
            neg = tf.gather(p, tf.reshape(tf.where(phi_p <= 0.0), (-1,)))
            p = tf.reshape(
                tf.sparse.to_dense(tf.sets.difference(tf.reshape(
                    p, (1, -1)), tf.reshape(neg, (1, -1)))),
                (-1,)
            )

            # look up the appropriate Sigma_prior indices to abandon given the
            # newly found negative pixel indices
            abandon_indices = tf.concat(
                tf.gather(abandon_lookup, neg), 0).flat_values
            Sigma_prior_retain = tf.tensor_scatter_nd_update(
                Sigma_prior_retain,
                tf.reshape(abandon_indices, (-1, 1)),
                tf.broadcast_to(False, tf.shape(abandon_indices)),
            )

            # return updated lop variables and and updated loop condition
            return [index+1, phi_p, p, Sigma_prior_retain, tf.greater(tf.shape(neg)[0], 0)]

        # initialize loop variables in the order that they appear in the update
        # arguments
        vars = [
            tf.constant(0, dtype=tf.int32),
            tf.zeros(self.G.shape[1], dtype=tf.float32),
            tf.convert_to_tensor(
                [i for i in range(0, self.G.shape[1])], dtype=tf.int32),
            tf.ones(len(self.Sigma_prior_values), dtype=tf.bool),
        ]

        # create expanded loop of update function that bypasses the function
        # once update returns a false loop condition
        cond = tf.constant(True, dtype=tf.bool)
        for _ in range(20):
            *vars, cond = tf.cond(
                cond,
                lambda: update(*vars),
                lambda: vars+[cond],
            )

        # publish useful results
        self.iters, self.phi_p, self.p, *_ = vars
