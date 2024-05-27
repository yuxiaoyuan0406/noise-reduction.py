import numpy as np
import util
import matplotlib.pyplot as plt
from numba import njit


@njit
def t_to_f(x: np.ndarray, dt: float):
    f, df = np.linspace(-.5 / dt,
                        .5 / dt,
                        len(x),
                        endpoint=False,
                        retstep=True)
    X = np.fft.fftshift(np.fft.fft(x))
    return f, df, X


@njit
def f_to_t(X: np.ndarray, df: float, t0: float = 0):
    t, dt = np.linspace(t0, t0 + 1 / df, len(X), endpoint=False, retstep=True)
    x = np.fft.ifft(np.fft.ifftshift(X))
    return t, dt, x


class Signal:
    """Signal generation class.
    """

    def __init__(
        self,
        val: np.ndarray,
        t=None,
        f=None,
        color=None,
        label: str = '',
    ):
        self.color = color
        self.label = label
        if t is None and f is not None:
            self.f = np.array(f)
            self.df = f[1] - f[0]
            self.X = np.array(val)
            self.t, self.dt, self.x = f_to_t(self.X, self.df, 0)
        elif t is not None and f is None:
            self.t = np.array(t)
            self.dt = t[1] - t[0]
            self.x = np.array(val)
            self.f, self.df, self.X = t_to_f(self.x, self.dt)
        else:
            return

    def plot_time_domain(self, ax=None, show=False, block=False):
        """Plot the signal in time domain

        Args:
            ax (matplotlib.axes.Axes, optional): The figure to plot on. Defaults to None.
            show (bool, optional): Show. Defaults to False.
            block (bool, optional): Block. Defaults to False.

        Returns:
            Matplotlib axes.
        """
        # 检查是否提供了Axes对象，如果没有则创建新的Figure和Axes
        if ax is None:
            fig, ax = plt.subplots()
            ax.legend(loc='upper right')
            ax.grid(True)
        else:
            fig = ax.figure  # 获取Axes所属的Figure对象

        ax.plot(self.t, self.x, color=self.color,
                label=self.label)  # 使用提供的x_data和y_data进行作图
        # ax.set_title('Data Plot')
        # ax.set_xlabel('X-Axis')
        # ax.set_ylabel('Y-Axis')

        plt.tight_layout()
        if show:
            plt.show(block=block)

        return ax  # 返回Axes对象

    def plot_freq_domain(self,
                         ax_power=None,
                         ax_phase=None,
                         show=False,
                         block=False):
        """Plot the signal in frequency domain

        Args:
            ax_power (matplotlib.axes.Axes, optional): The axe to plot power. Defaults to None.
            ax_phase (matplotlib.axes.Axes, optional): The axe to plot phase. Defaults to None.
            show (bool, optional): Show. Defaults to False.
            block (bool, optional): Block. Defaults to False.

        Returns:
            Matplotlib axes.
        """
        if ax_power is None or ax_phase is None:
            fig, (ax_power, ax_phase) = plt.subplots(2, 1, sharex=True)
            ax_power.grid(True)
            ax_phase.grid(True)
            ax_power.legend(loc='upper right')
            ax_phase.legend(loc='upper right')
            plt.tight_layout()
        else:
            fig = ax_power.figure

        ax_power.plot(self.f,
                      np.abs(self.X),
                      color=self.color,
                      label=self.label)
        ax_phase.plot(self.f,
                      np.angle(self.X, deg=True),
                      color=self.color,
                      label=self.label)

        if show:
            plt.show(block=block)

        return ax_power, ax_phase


class Filter:

    def __init__(
        self,
        func=None,
        color=None,
        label: str = '',
    ):
        self.filter = func
        self.color = color
        self.label = label

    def apply(
        self,
        sig: Signal,
        color=None,
        label: str = '',
    ):
        filt = self.filter(sig.f)
        out = Signal(filt * sig.X, f=sig.f, color=color, label=label)
        return out

if __name__ == '__main__':
    pass
