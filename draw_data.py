import numpy as np
from module import *

def load_from_text(filename, label='data'):
    data = np.loadtxt(filename).transpose()
    t = data[0]
    val = data[1]
    return Signal(val, t=t, label=label)

if __name__ == '__main__':
    ax_time, ax_power, ax_phase = None, None, None

    sig = load_from_text('data/4/VCOM.matlab', label='signal')
    # pos = load_from_text('data/4/out-pos.matlab', label='pos')
    # neg = load_from_text('data/4/out-neg.matlab', label='neg')

    # ax_time = pos.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = pos.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)
    # ax_time = neg.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = neg.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)


    # diff = Signal(pos.x + neg.x, t=pos.t, label='diff')


    # ax_time = diff.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = diff.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    def band_pass_filter_func(f, low, high):
        val = []
        for _ in f:
            if low <= np.abs(_) and np.abs(_) <= high:
                val.append(1.)
            else:
                val.append(0.)
        return np.array(val, dtype=np.float64)
    def func(f):
        return band_pass_filter_func(f, 10, 10 * 1e3)

    band_filter = Filter(func, label='band pass')
    # band_filter.plot(pos.f)

    # pos_bandfilt = band_filter.apply(pos, label='pos->band pass')
    # neg_bandfilt = band_filter.apply(neg, label='neg->band pass')
    sig_bandfilt = band_filter.apply(sig, label='sig->band pass')


    ax_time, ax_power, ax_phase = None, None, None
    # ax_time = pos_bandfilt.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = pos_bandfilt.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)
    # ax_time = neg_bandfilt.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = neg_bandfilt.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)
    ax_time = sig_bandfilt.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = sig_bandfilt.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)


    plt.show()
    # input('Enter to exit...')
    # plt.close('all')
