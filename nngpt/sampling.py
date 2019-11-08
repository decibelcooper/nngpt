import logging
import numpy as np
import time


def sample_normal(means, covs, n_samples):
    t0 = time.time()

    xy = np.concatenate([
        np.random.multivariate_normal(
            mean, cov, int(n_sample),
        ) for mean, cov, n_sample in zip(means, covs, n_samples)
    ])

    t1 = time.time()
    logging.info(f'sampling time: {t1-t0:.3f}')

    return xy


def sample_circle(n=1e7, r=10, sigma=2):
    t0 = time.time()

    phi = np.random.uniform(0, 2*np.pi, int(n))
    cx = r*np.cos(phi)
    cy = r*np.sin(phi)

    xy = np.transpose([cx, cy]) + \
        np.random.multivariate_normal(
            [0.0, 0.0], [[sigma, 0.0], [0.0, sigma]], len(phi))

    t1 = time.time()
    logging.info(f'sampling time: {t1-t0:.3f}')

    return xy


def sample_square(n=1e7, a=15.708, sigma=2):
    t0 = time.time()

    choices = [-a/2, a/2]
    xy = np.concatenate([
        np.transpose([np.random.uniform(*choices, int(n/2)),
                      np.random.choice(choices, int(n/2))]) +
        np.random.multivariate_normal(
            [0.0, 0.0], [[sigma, 0.0], [0.0, sigma]], int(n/2)),
        np.transpose([np.random.choice(choices, int(n/2)),
                      np.random.uniform(*choices, int(n/2))]) +
        np.random.multivariate_normal(
            [0.0, 0.0], [[sigma, 0.0], [0.0, sigma]], int(n/2)),
    ])

    t1 = time.time()
    logging.info(f'sampling time: {t1-t0:.3f}')

    return xy


def add_diffusion(xy, max_diff_sigma):
    t0 = time.time()

    xy_diff = xy + np.random.multivariate_normal([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], len(xy)) * \
        np.reshape(max_diff_sigma *
                   np.sqrt(np.random.uniform(0, 1, len(xy))), (-1, 1))

    t1 = time.time()
    logging.info(f'sampling time: {t1-t0:.3f}')

    return xy_diff
