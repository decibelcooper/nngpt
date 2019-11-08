import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch
import matplotlib.transforms as transforms


def confidence_ellipse(mean, cov, ax, n_std=3.0, facecolor='none', **kwargs):
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = pltpatch.Ellipse((0, 0),
                               width=ell_radius_x * 2,
                               height=ell_radius_y * 2,
                               facecolor=facecolor,
                               **kwargs)

    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean[0], mean[1])

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def draw_tomo(
    img, width, height, colorbar=False, cmap='YlGnBu',
    mean=None, cov=None, true_mean=None, true_cov=None, n_std=4,
):
    if not isinstance(img, list):
        img = [img]

    fig, axs = plt.subplots(1, len(img), sharex=True, sharey=True)
    if len(img) > 1:
        fig.set_figheight(6)
        fig.set_figwidth(8)
    vmin = np.min(img)
    vmax = np.max(img)

    if not isinstance(axs, np.ndarray):
        axs = [axs]

    for i, ax in enumerate(axs):
        im = ax.imshow(
            img[i],
            extent=(
                width/2,
                -width/2,
                -height/2,
                height/2,
            ),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.label_outer()

        if mean is not None and cov is not None:
            if mean[i] is not None and cov[i] is not None:
                for m, c in zip(mean[i], cov[i]):
                    confidence_ellipse(
                        m, c, ax, n_std=n_std, edgecolor='red')
        if true_mean is not None and true_cov is not None:
            if true_mean[i] is not None and true_cov[i] is not None:
                for m, c in zip(true_mean[i], true_cov[i]):
                    confidence_ellipse(
                        m, c, ax, n_std=n_std, edgecolor='cyan')

    if colorbar:
        if len(img) > 1:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.3, 0.025, 0.4])
            fig.colorbar(im, cax=cbar_ax)
        else:
            fig.colorbar(im)

    fig.show()
