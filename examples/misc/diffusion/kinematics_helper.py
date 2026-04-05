from collections import OrderedDict
from matplotlib import cm, patches
import matplotlib.pyplot as plt
import numpy as np

import scipy
from sklearn.cluster import MeanShift
from sklearn.neighbors import KernelDensity


class InverseKinematicsModel:
    n_parameters = 4
    n_observations = 2
    name = "inverse-kinematics"

    def __init__(self, lens=[0.5, 0.5, 1.0], sigmas=[0.25, 0.5, 0.5, 0.5], linecolors=["gray"] * 3):
        self.name = "inverse-kinematics"
        self.lens = np.array(lens)
        self.sigmas = np.array(sigmas)
        self.rangex = (-0.35, 2.25)
        self.rangey = (-1.3, 1.3)

        cmap = cm.tab20c
        self.colors = [[cmap(4 * c_index), cmap(4 * c_index + 1), cmap(4 * c_index + 2)] for c_index in range(5)][-1]
        self.linecolors = linecolors

        self.prior_reference = OrderedDict(
            theta_1=scipy.stats.norm(0, sigmas[0]),
            theta_2=scipy.stats.norm(0, sigmas[1]),
            theta_3=scipy.stats.norm(0, sigmas[2]),
            theta_4=scipy.stats.norm(0, sigmas[3]),
        )

    def sample_prior(self, N):
        return np.random.randn(N, 4) * self.sigmas

    @staticmethod
    def segment_points(p_, length, angle):
        p = np.array(p_)
        angle = np.array(angle)
        p[:, 0] += length * np.cos(angle)
        p[:, 1] += length * np.sin(angle)
        return p_, p

    def forward_process(self, x):
        start = np.stack([np.zeros((x.shape[0])), x[:, 0]], axis=1)
        _, x1 = self.segment_points(start, self.lens[0], x[:, 1])
        _, x2 = self.segment_points(x1, self.lens[1], x[:, 1] + x[:, 2])
        _, y = self.segment_points(x2, self.lens[2], x[:, 1] + x[:, 2] + x[:, 3])
        return y

    @staticmethod
    def find_MAP(x):
        mean_shift = MeanShift()
        mean_shift.fit(x)
        centers = mean_shift.cluster_centers_
        kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(x)

        best_center = (None, -np.inf)
        dens = kde.score_samples(centers)
        for c, d in zip(centers, dens):
            if d > best_center[1]:
                best_center = (c.copy(), d)

        dist_to_best = np.sum((x - best_center[0]) ** 2, axis=1)
        return np.argmin(dist_to_best)

    @staticmethod
    def arcarrow(
        start,
        target,
        dist=0.3,
        open_angle=150,
        kw=dict(arrowstyle="<->, head_width=1, head_length=2", ec="black", lw=0.5),
    ):
        direction = target - start
        angle = np.arctan2(direction[1], direction[0])

        angle1 = angle - np.radians(open_angle / 2)
        x1 = start[0] + dist * np.cos(angle1)
        y1 = start[1] + dist * np.sin(angle1)
        angle2 = angle + np.radians(open_angle / 2)
        x2 = start[0] + dist * np.cos(angle2)
        y2 = start[1] + dist * np.sin(angle2)

        plt.gca().add_patch(patches.FancyArrowPatch((x1, y1), (x2, y2), connectionstyle="arc3, rad=.6", **kw))

    def draw_isolines(self, samples, color, filter_width, ax=None):
        if not filter_width > 0:
            return

        x = np.array(samples)

        starting_pos = np.zeros((x.shape[0], 2))
        starting_pos[:, 1] = x[:, 0]

        x0, x1 = self.segment_points(starting_pos, self.lens[0], x[:, 1])
        x1, x2 = self.segment_points(x1, self.lens[1], x[:, 1] + x[:, 2])
        x2, y = self.segment_points(x2, self.lens[2], x[:, 1] + x[:, 2] + x[:, 3])

        hist, xbins, ybins = np.histogram2d(y[:, 0], y[:, 1], bins=600, range=[self.rangex, self.rangey], density=True)
        hist = scipy.ndimage.gaussian_filter(hist, filter_width)

        percentile = 0.03 * np.sum(hist)
        for q in np.logspace(-99, np.log10(np.max(hist)), 8000, endpoint=True):
            if np.sum(hist[hist < q]) > percentile:
                break
        else:
            q = 1.0

        X, Y = np.meshgrid(0.5 * (xbins[:-1] + xbins[1:]), 0.5 * (ybins[:-1] + ybins[1:]))

        if ax is None:
            plt.contour(X, Y, hist.T, [q], colors=color, linewidths=0.7, zorder=3)
        else:
            ax.contour(X, Y, hist.T, [q], colors=color, linewidths=0.7, zorder=3)

    @staticmethod
    def init_plot():
        return plt.figure(figsize=(8, 8))

    def update_plot_ax(
        self,
        ax,
        x,
        y_target,
        exemplar=None,
        arrows=False,
        target_label=False,
        vline_color="black",
        exemplar_color="#e6e7eb",
        cross_color="maroon",
    ):
        x = np.array(x)  # [:4000, :]
        if exemplar is None:
            exemplar = self.find_MAP(x)

        starting_pos = np.zeros((x.shape[0], 2))
        starting_pos[:, 1] = x[:, 0]
        x0, x1 = self.segment_points(starting_pos, self.lens[0], x[:, 1])
        x1, x2 = self.segment_points(x1, self.lens[1], x[:, 1] + x[:, 2])
        x2, x3 = self.segment_points(x2, self.lens[2], x[:, 1] + x[:, 2] + x[:, 3])

        ax.axvline(x=0, c=vline_color, linewidth=1, alpha=0.8)

        if not arrows:
            l_cross = 0.6
            ax.plot(
                [y_target[0] - l_cross, y_target[0] + l_cross],
                [y_target[1], y_target[1]],
                ls="-",
                c=cross_color,
                linewidth=0.8,
                alpha=1.0,
                rasterized=True,
            )  # , zorder=-1)
            ax.plot(
                [y_target[0], y_target[0]],
                [y_target[1] - l_cross, y_target[1] + l_cross],
                ls="-",
                c=cross_color,
                linewidth=0.8,
                alpha=1.0,
                rasterized=True,
            )  # , zorder=-1)
            if target_label:
                ax.text(
                    y_target[0] + 0.15,
                    y_target[1] + 0.15,
                    target_label,
                    ha="left",
                    va="bottom",
                    color="magenta",
                    fontsize=10,
                )

        opts = {
            "alpha": 0.10,
            "scale": 1,
            "angles": "xy",
            "scale_units": "xy",
            "headlength": 0,
            "headaxislength": 0,
            "linewidth": 1.0,
            "rasterized": True,
        }
        ax.quiver(x0[:, 0], x0[:, 1], (x1 - x0)[:, 0], (x1 - x0)[:, 1], **{"color": self.linecolors[0], **opts})
        ax.quiver(x1[:, 0], x1[:, 1], (x2 - x1)[:, 0], (x2 - x1)[:, 1], **{"color": self.linecolors[1], **opts})
        ax.quiver(x2[:, 0], x2[:, 1], (x3 - x2)[:, 0], (x3 - x2)[:, 1], **{"color": self.linecolors[2], **opts})
        ax.scatter(x3[:, 0], x3[:, 1], color=self.linecolors[0], s=1, rasterized=True, alpha=0.20)

        ax.plot(
            [x0[exemplar, 0], x1[exemplar, 0], x2[exemplar, 0]],
            [x0[exemplar, 1], x1[exemplar, 1], x2[exemplar, 1]],
            "-",
            color=exemplar_color,
            linewidth=2,
            zorder=4,
            rasterized=True,
        )

        if arrows:
            ax.annotate(
                s="",
                xy=(-0.125, -0.5),
                xytext=(-0.125, 0.5),
                arrowprops=dict(arrowstyle="<->, head_width=.1, head_length=.2", ec="black", lw="0.5"),
                zorder=2,
            )
            self.arcarrow(x0[exemplar, :], x1[exemplar, :])
            self.arcarrow(x1[exemplar, :], x2[exemplar, :])
            self.arcarrow(x2[exemplar, :], x3[exemplar, :])
            ax.text(-0.09, -0.60, r"$x_1$", ha="center", va="center", fontsize=10)
            ax.text(0.13, -0.38, r"$x_2$", ha="center", va="center", fontsize=10)
            ax.text(0.60, -0.40, r"$x_3$", ha="center", va="center", fontsize=10)
            ax.text(1.10, -0.44, r"$x_4$", ha="center", va="center", fontsize=10)
            ax.text(1.97, -0.27, r"$\mathbf{y}$", ha="center", va="center", fontsize=10)

        ax.arrow(
            x2[exemplar, 0],
            x2[exemplar, 1],
            x3[exemplar, 0] - x2[exemplar, 0],
            x3[exemplar, 1] - x2[exemplar, 1],
            color=exemplar_color,
            linewidth=2,
            head_width=0.05,
            head_length=0.04,
            overhang=0.1,
            length_includes_head=True,
            zorder=4,
        )
        ax.scatter(
            [
                x0[exemplar, 0],
            ],
            [
                x0[exemplar, 1],
            ],
            s=40,
            marker="s",
            linewidth=1,
            edgecolors="black",
            facecolors=exemplar_color,
            alpha=0.8,
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            [x0[exemplar, 0], x1[exemplar, 0], x2[exemplar, 0]],
            [x0[exemplar, 1], x1[exemplar, 1], x2[exemplar, 1]],
            s=20,
            linewidth=1,
            edgecolors="black",
            facecolors=exemplar_color,
            alpha=0.8,
            zorder=5,
            rasterized=True,
        )

        ax.set_xlim(-0.01, 1.8)
        ax.set_ylim(*self.rangey)
        ax.set_aspect("equal")

        ax.set_xticks([])
        ax.set_yticks([])

        return ax
