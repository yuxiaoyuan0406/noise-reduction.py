import matplotlib.pyplot as plt
import numpy as np
import pyfftw
# from numba import njit

EPSILON = 1e-12 # To avoid calculate log of zero

def fft(x: np.ndarray):
    # Create arrays aligned for FFTW
    x_aligned = pyfftw.empty_aligned(x.shape, dtype='complex128')
    X_aligned = pyfftw.empty_aligned(x.shape, dtype='complex128')

    # Initialize FFT object
    fft_object = pyfftw.FFTW(x_aligned, X_aligned)

    # Copy data to FFTW input
    x_aligned[:] = x

    # Compute FFT and shift
    fft_object()  # FFT is performed in-place
    X = np.fft.fftshift(X_aligned)

    return X


def ifft(X: np.ndarray):
    # Create arrays aligned for FFTW
    X_aligned = pyfftw.empty_aligned(X.shape, dtype='complex128')
    x_aligned = pyfftw.empty_aligned(X.shape, dtype='complex128')

    # Initialize IFFT object
    ifft_object = pyfftw.FFTW(X_aligned, x_aligned, direction='FFTW_BACKWARD')

    # Copy data to FFTW input
    X_aligned[:] = X

    # Compute IFFT and shift
    ifft_object()  # IFFT is performed in-place
    x = np.fft.ifftshift(x_aligned)

    return x


# @njit
def t_to_f(x: np.ndarray, dt: float):
    """transform signal from time domain to frequency domain

    Args:
        x (np.ndarray): input data in time domain
        dt (float): time resolution

    Returns:
        np.ndarray, float, np.ndarray: frequency array, frequency resolution, signal in frequency domain
    """
    f, df = np.linspace(-.5 / dt,
                        .5 / dt,
                        len(x),
                        endpoint=False,
                        retstep=True)
    X = np.fft.fftshift(np.fft.fft(x))
    # X = np.fft.fftshift(fft(x))
    return f, df, X


# @njit
def f_to_t(X: np.ndarray, df: float, t0: float = 0):
    """transform signal from frequency domain to time domain

    Args:
        X (np.ndarray): input data in frequency domain
        df (float): frequency resolution
        t0 (float, optional): the begin time of time array. Defaults to 0.

    Returns:
        np.ndarray, float, np.ndarray: time sequence, time resolution, data in time domain
    """
    t, dt = np.linspace(t0, t0 + 1 / df, len(X), endpoint=False, retstep=True)
    x = np.fft.ifft(np.fft.ifftshift(X))
    # x = ifft(np.fft.ifftshift(X))
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
            if len(val) != len(f):
                raise ValueError(
                    f'`val` and `f` must have same length, but have length {len(val)} and {len(f)}'
                )
            self.f = np.array(f)
            self.df = f[1] - f[0]
            self.X = np.array(val)
            self.t, self.dt, self.x = f_to_t(self.X, self.df, 0)
            # self.x = np.real(self.x)
        elif t is not None and f is None:
            if len(val) != len(t):
                raise ValueError(
                    f'`val` and `t` must have same length, but have length {len(val)} and {len(t)}'
                )
            self.t = np.array(t)
            self.t = self.t - self.t[0]
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
            ax.grid(True)
        else:
            fig = ax.figure  # 获取Axes所属的Figure对象

        ax.plot(self.t, np.real(self.x), color=self.color,
                label=self.label)  # 使用提供的x_data和y_data进行作图
        # ax.set_title('Data Plot')
        # ax.set_xlabel('X-Axis')
        # ax.set_ylabel('Y-Axis')
        ax.legend(loc='upper right')

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
            # ax_power.set_xscale('log')
            # ax_phase.set_xscale('log')
            ax_power.set_ylabel('dB')
            plt.tight_layout()
        else:
            fig = ax_power.figure

        # calculate white noise to avoid log(0) or arctant(y/0)
        non_zero_min = np.min(np.abs(self.X[np.abs(self.X) > 0]))
        mean_amplitude = np.mean(np.abs(self.X))
        epsilon = min(non_zero_min * 0.01, mean_amplitude * 0.01, np.finfo(self.X.dtype).tiny)        
        # epsilon = EPSILON

        ax_power.plot(
            self.f,
            20 * np.log10(np.abs(self.X) + epsilon),
            # np.abs(self.X),
            color=self.color,
            label=self.label)
        ax_phase.plot(self.f,
                      np.angle(self.X + epsilon, deg=True),
                      color=self.color,
                      label=self.label)

        ax_power.legend(loc='upper right')
        ax_phase.legend(loc='upper right')

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
        """Apply filter to target signal in frequency domain.

        Args:
            sig (Signal): Input signal
            color (_type_, optional): The color of output signal when plotting. Defaults to None.
            label (str, optional): The label of the output signal. Defaults to ''.

        Returns:
            Signal: Output.
        """
        filt = self.filter(sig.f)
        out = Signal(filt * sig.X, f=sig.f, color=color, label=label)
        return out

    def plot(
        self,
        f,
        ax_power=None,
        ax_phase=None,
        show=False,
        block=False,
    ):
        """Plot the frequency figure of the filter.

        Args:
            f (np.ndarray): The frequency sequency to plot.
            ax_power (matplotlib.axes.Axes, optional): The axe to plot power. Defaults to None.
            ax_phase (matplotlib.axes.Axes, optional): The axe to plot phase. Defaults to None.
            show (bool, optional): Show. Defaults to False.
            block (bool, optional): When show, block. Defaults to False.

        Returns:
            axes: Axes.
        """
        sig = Signal(np.ones(f.shape) * self.filter(f), f=f, label=self.label)
        return sig.plot_freq_domain(ax_power=ax_power,
                                    ax_phase=ax_phase,
                                    show=show,
                                    block=block)

def load_from_text(filename, label='data'):
    data = np.loadtxt(filename).transpose()
    t = data[0]
    val = data[1]
    return Signal(val, t=t, label=label)


class WhiteNoise(Signal):

    def __init__(
        self,
        t: np.ndarray,
        loc: float = 0,
        scale: float = 1,
        color=None,
        label: str = 'White Noise',
    ):
        noise = np.random.normal(loc=loc, scale=scale, size=len(t))
        super().__init__(val=noise, t=t, color=color, label=label)


class SquareWave(Signal):

    def __init__(self,
                 t: np.ndarray,
                 freq: float = 1,
                 amp: float = 1,
                 init_phase: float = 0,
                 bias: float = 0,
                 ideal=False,
                 color=None,
                 label: str = 'Square Wave'):
        period = 1 / freq
        dt = t[1] - t[0]
        if ideal:
            # val = []
            period = int(period / dt)
            # for _ in range(len(t)):
            val = np.array(
                [1 if _ % period < period / 2 else -1 for _ in range(len(t))],
                dtype=np.float64)
        else:
            val = np.sign(np.sin(2 * np.pi * freq * t + init_phase))
        val = val * amp + bias
        super().__init__(val, t=t, color=color, label=label)
