import numpy as np
import matplotlib.pyplot as plt
import util
from module import *


if __name__ == '__main__':
    ## plot time domain
    ax_time = None
    ## plot frequency domain
    ax_power, ax_phase = None, None

    ## simulation parameters
    t0 = 0
    dt = 1e-8
    t_len = .05
    t = np.linspace(t0, t0 + t_len, int(t_len/dt), endpoint=False)

    ## original signal
    f0 = 40 * 1e3
    original = Signal(np.sin(2 * np.pi * f0 * t), t=t, label='original signal')
    # ax_time = original.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = original.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## PID
    def pid_func(f):
        # w = 2 * np.pi * f
        # return ((1j) * w) / (1 + (1j) * w)
        return ((1j) * f / (50 * 1e3)) / (1 + (1j) * f / (50 * 1e3))
    ## compare signal
    pid = Filter(pid_func, label='PID module')
    # pid.plot(original.f, ax_power=ax_power, ax_phase=ax_phase)
    original_pid = pid.apply(original, label='Original After PID')
    # ax_time = original_pid.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = original_pid.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    compare = Signal(0.6 * np.sin(2 * np.pi * f0 * t + 0 * np.pi), t=t, label='campare')
    # ax_time = compare.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = compare.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## sample original signal
    smp = []
    for i in range(len(original.t)):
        if i % 64 < 32:
            smp.append(original.x[i])
        else:
            smp.append(smp[i-1])
    original_sampled = Signal(smp, t=t, label='Sampled signal')
    # ax_time = original_sampled.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = original_sampled.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## After sampled, go through PID
    smp_pid = pid.apply(original_sampled, label='Sampled then PID')
    # ax_time = smp_pid.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = smp_pid.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## Low pass
    def lp_filter(f):
        return util.low_pass_filter(f, 500 * 1e3)
    lp = Filter(lp_filter, label='Low pass filter')
    # lp.plot(original.f, ax_power=ax_power, ax_phase=ax_phase)
    smp_pid_lp = lp.apply(smp_pid, label='Sampled->PID->Low Pass')
    # ax_time = smp_pid_lp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = smp_pid_lp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    t_smp, smp = util.sample(smp_pid_lp.x, smp_pid_lp.t, 64, 20)
    smp_pid_lp_smp = Signal(smp, t=t_smp, label='Smp->PID->LP->Smp')
    # ax_time = smp_pid_lp_smp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = smp_pid_lp_smp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## Go through high pass filter
    def hp_filter(f):
        return util.high_pass_filter(f, 10 * 1e3, 50 * 1e3)
    hp = Filter(hp_filter, label='High pass filter')
    # hp.plot(original.f, ax_power=ax_power, ax_phase=ax_phase)
    sampled_hp = hp.apply(original_sampled, label='Sampled then high pass')
    # ax_time = sampled_hp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = sampled_hp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    smp_hp_lp = lp.apply(sampled_hp, label='Smp->HP->LP')
    # ax_time = smp_hp_lp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = smp_hp_lp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## sample after high pass filter
    smp = []
    t_smp = []
    t_smp, smp = util.sample(smp_hp_lp.x, smp_hp_lp.t, 64, 20)
    # for i in range(len(sampled_hp.t)):
    #     if i % 32 == 11:
    #         t_smp.append(sampled_hp.t[i])
    #         smp.append(sampled_hp.x[i])
    smp_hp_smp = Signal(smp, t=t_smp, label='Smp->HP->LP->Smp')
    ax_time = smp_hp_smp.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = smp_hp_smp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## Show plot
    plt.show(block=False)
    
    ## close all
    input("Enter to exit...")
    plt.close('all')

