import numpy as np
import matplotlib.pyplot as plt
import util
from module import *

def system1(input:Signal):
    ## Low pass
    def lp_filter(f):
        # return 1
        return util.ideal_low_pass_filter(f, 1 * 1e6)
    lp = Filter(lp_filter, label='Low pass filter')
    # lp.plot(input.f)

    ## High pass
    def hp_filter(f):
        return 1
        return util.high_pass_filter(f, 10 * 1e3, 500 * 1e3)
    hp = Filter(hp_filter, label='High pass filter')
    # hp.plot(original.f, ax_power=ax_power, ax_phase=ax_phase)

    x1 = hp.apply(input, label=f'{input.label}->HP')
    x1.plot_time_domain()
    x1.plot_freq_domain()
    x2 = lp.apply(x1, label=f'{x1.label}->LP')
    x2.plot_time_domain()
    x2.plot_freq_domain()
    # plt.show(block=False)
    t_out, out = util.sample(x2.x, x2.t, 512, 200)
    out = Signal(out, t=t_out, label=f'{x2.label}->Smp')
    return out

if __name__ == '__main__':
    ## plot time domain
    ax_time = None
    ## plot frequency domain
    ax_power, ax_phase = None, None

    ## simulation parameters
    t0 = 0
    dt = 1.25e-7 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5 * 0.5
    # dt = 1e-6
    t_len = .25 * 0.5 * 0.5 * 0.5
    t = np.linspace(t0, t0 + t_len, int(t_len/dt), endpoint=False)

    ## original signal
    f0 = 10 * 1e3
    original = Signal(np.sin(2 * np.pi * f0 * t), t=t, label='original')
    ax_time = original.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = original.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

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
    period = 512
    for i in range(len(original.t)):
        if i % period < period/2:
            # if i % period == 0 and i // period != 0:
            #     smp.append((original.x[i] + smp[i-1])/2)
            # else:
                smp.append(original.x[i])
        else:
            smp.append(smp[i-1])
    original_sampled = Signal(smp, t=t, label='Ori->Smp')
    # ax_time = original_sampled.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = original_sampled.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    # ori_smp_hp_lp = system1(original_sampled)
    # ax_time = ori_smp_hp_lp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = ori_smp_hp_lp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)
    ori_hp_lp = system1(original)
    ax_time = ori_hp_lp.plot_time_domain(ax=ax_time)
    ax_power, ax_phase = ori_hp_lp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

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
    # ax_time = smp_hp_smp.plot_time_domain(ax=ax_time)
    # ax_power, ax_phase = smp_hp_smp.plot_freq_domain(ax_power=ax_power, ax_phase=ax_phase)

    ## Show plot
    plt.show(block=False)
    
    ## close all
    input("Enter to exit...")
    plt.close('all')

