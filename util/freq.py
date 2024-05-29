'''
Functions for run a fft and plot the Bode diagram.
'''
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
from numba import njit

matplotlib.use('TkAgg')


def power_and_phase(
    a: np.ndarray,
    dt: float,
    log: bool = False,
    max_freq=None,
):
    '''
    Calculate the amplitude spectrum and phase spectrum 
    of a sequence given a sampling period.
    '''
    _f_a = np.fft.fftshift(np.fft.fft(a))
    f_a_power = np.abs(_f_a)
    f_a_phase = np.angle(_f_a)
    n = len(a)
    f = np.linspace(-.5 / dt, .5 / dt, n)

    if max_freq is not None:
        valid_indices = np.abs(f) <= max_freq
        if log:
            positive_indices = f > 0
            valid_indices = [
                valid_indices[i] and positive_indices[i]
                for i in range(len(valid_indices))
            ]
        f = f[valid_indices]
        f_a_power = f_a_power[valid_indices]
        f_a_phase = f_a_phase[valid_indices]

    f_a_phase = np.unwrap(f_a_phase)
    if log:
        f_a_power = 20 * np.log10(f_a_power)

    return f, _f_a, f_a_power, f_a_phase


def freq_and_plot(
    a: np.ndarray,
    dt: float,
    label: str = '',
    log: bool = False,
    max_freq=None,
    show=False,
    block=False,
    power_ax=None,
    phase_ax=None,
):
    '''
    Calculate the spectrum of the sequence given a sampling period 
    and plot the Bode diagram.
    Return the axes that can draw more onto.
    '''
    f, _, f_a_power, f_a_phase = power_and_phase(a, dt, log, max_freq)
    # window_name='fft'

    if power_ax is None or phase_ax is None:
        fig, (power_ax, phase_ax) = plt.subplots(2, 1, sharex=True)

    power_ax.plot(f, f_a_power, label=label)
    # power.xlabel('f(Hz)')
    # power.grid(True)
    if log:
        power_ax.set_xscale('log')

    phase_ax.plot(f, f_a_phase, label=label)
    # phase.xlabel('f(Hz)')
    # phase.grid(True)
    if log:
        phase_ax.set_xscale('log')
    phase_ax.yaxis.set_major_locator(MultipleLocator(base=np.pi))
    phase_ax.yaxis.set_major_formatter(
        FuncFormatter(lambda val, pos: f'{val / np.pi:.2f}π'))

    plt.tight_layout()
    # plt.grid()
    if show:
        plt.show(block=block)

    return power_ax, phase_ax

@njit
def high_pass_filter(f: np.ndarray, fz: float, fp: float):
    '''
    Return a high pass filter with given `f_z` and `f_p`.
    ```
    H(f) = (1 + j f/f_z) / (1 + j f/f_p)
    ```
    '''
    return (1 + (1j) * f / fz) / (1 + (1j) * f / fp)    

@njit
def ideal_low_pass_filter(f:np.ndarray, f0: float):
    _ = np.ones(f.shape)
    for i in range(len(f)):
        if np.abs(f[i]) >= f0:
            _[i] = 0
    return _

@njit
def low_pass_filter(f, f0):
    return 1 / (1 + (1j) * f / f0)

# @njit
def t_to_f(x: np.ndarray, dt: float, retstep=False):
    f, df = np.linspace(-.5/dt, .5/dt, len(x), endpoint=False, retstep=True)
    X = np.fft.fftshift(np.fft.fft(x))
    if retstep:
        return f,df,X
    else:
        return f, X

# @njit
def f_to_t(X: np.ndarray, df: float, t0: float = 0, retstep=False):
    t, dt = np.linspace(t0, t0 + 1/df, len(X), endpoint=False, retstep=True)
    x = np.fft.ifft(np.fft.ifftshift(X))
    if retstep:
        return t, dt, x
    else:
        return t, x

@njit
def difference(x: np.ndarray, t: np.ndarray):
    diff = x[1:] - x[:-1]
    return t[:-1], diff

@njit
def delete_picked(x:np.ndarray, gap: int, bias: int):
    for _ in range(len(x)):
        if _ % gap == bias:
            x[_] = x[_-1]
    return x

@njit
def sample(x: np.ndarray, t: np.ndarray, gap: int, bias: int):
    t_sp = []
    x_sp = []
    for _ in range(len(t)):
        if _ % gap == bias:
            t_sp.append(t[_])
            x_sp.append(x[_])
    return np.array(t_sp), np.array(x_sp)

@njit
def sample_t(x: np.ndarray, t: np.ndarray, ts: float, bias: float):
    dt = t[1] - t[0]
    gap = int(ts/dt)
    bias = int(bias/dt)
    return sample(x, t, gap, bias)

if __name__ == '__main__':
    t,dt = np.linspace(0,2,2000000, endpoint=False, retstep=True)
    a = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*5e4*t)
    # a2=np.sin(2*np.pi*100*t) + np.sin(2*np.pi*5e8*t)
    # disp = np.loadtxt('disp_c.dat').transpose()[2]
    # disp = np.loadtxt('bit_c.dat')

    # freq_and_plot(disp, 5e-7, log=True, max_freq=1e4)
    # freq_and_plot(a, dt)
    # freq_and_plot(np.array([a,a2]), dt)



    plt.figure()
    plt.plot(t, a)
    plt.show(block=False)

    input('Press Enter to exit...')
    plt.close('all')
