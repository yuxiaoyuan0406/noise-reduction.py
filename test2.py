import numpy as np
import matplotlib.pyplot as plt
import util
from numba import njit

@njit
def chop_gen(t: np.ndarray, gap: int):
    chop = []
    for _ in range(len(t)):
        if _ % gap < gap / 2:
            chop.append(1)
        else:
            # chop.append(-1)
            chop.append(0)
    return np.array(chop)

@njit
def de_mod(x: np.ndarray, chop: np.ndarray):
    assert len(x) == len(chop)
    de = []
    for _ in range(len(x)):
        if chop[_] > 0:
            this = x[_]
        else:
            this = de[_-1]
        de.append(this)
    return np.array(de)
            
if __name__ == '__main__':
    dt = 5e-7
    runtime = 2.
    length = int(runtime / dt)
    t, dt = np.linspace(0,runtime,length, retstep=True, endpoint=False)

    # original signal
    f0 = 50
    x = np.sin(2 * np.pi * f0 * t)# + 1e-3 * np.cos(2 * np.pi * f0 * t)

    ## Frequency domain of original signal
    f, df, X = util.t_to_f(x, dt, retstep=True)

    chop = chop_gen(t, 32)

    # chop original signal
    x_chopped = x * chop
    
    # sample
    x_smp = []
    for _ in range(len(x)):
        if _ % 16 == 0:
            x_smp.append(x[_])
        else:
            x_smp.append(x_smp[_-1])
    x_smp = np.array(x_smp)

    # chop sampled signal
    x_smp_chopped = x_smp * chop
    # noise
    noise = np.sin(2 * np.pi * 1e2 * t) + np.sin(2 * np.pi * 60 * t) + np.sin(2 * np.pi * 49 * t)
    x_smp_chopped += noise

    t_cds = []
    x_smp_chopped_cds = []
    for _ in range(len(x_smp_chopped)):
        if _ % 32 == 0:
            try:
                x_smp_chopped_cds.append(x_smp_chopped[_] - x_smp_chopped[_+16])
                t_cds.append(t[_])
            except:
                pass
    x_smp_chopped_cds = np.array(x_smp_chopped_cds)

    # frequency domain of chopped signal
    f, df, X_chopped = util.t_to_f(x_chopped, dt, retstep=True)
    _, _,  X_smp_chopped = util.t_to_f(x_smp_chopped, dt, retstep=True)
    f_cds, df_cds, X_smp_chopped_cds = util.t_to_f(x_smp_chopped_cds, t_cds[1] - t_cds[0], retstep=True)

    # High pass filter
    hp = 1e-1 * util.high_pass_filter(f, 1e3, 10 * 1e3)
    X_chopped_hp = X_chopped * hp
    X_smp_chopped_hp = X_smp_chopped * hp
    t, dt, x_chopped_hp = util.f_to_t(X_chopped_hp, df, retstep=True)
    _, __, x_smp_chopped_hp = util.f_to_t(X_smp_chopped_hp, df, retstep=True)

    # differenct between pre-high-pass and pro-high-pass
    # X_chopped_hp_diff = X_chopped_hp - X_chopped
    # _, _, x_chopped_hp_diff = util.f_to_t(X_chopped_hp_diff, df, retstep=True)

    # demodulation
    # x_chopped_hp_demod = x_chopped_hp * chop
    x_chopped_hp_demod = de_mod(x_chopped_hp, chop)
    _, _, X_chopped_hp_demod = util.t_to_f(x_chopped_hp_demod, dt, retstep=True)
    lp = np.abs(util.low_pass_filter(f, 1e3))
    
    # Low pass
    X_chopped_hp_demod_lp = lp * X_chopped_hp_demod
    _, _, x_chopped_hp_demod_lp = util.f_to_t(X_chopped_hp_demod_lp, df, retstep=True)

    # difference
    t_diff, x_chopped_hp_demod_diff = util.difference(x_chopped_hp_demod, t)
    # x_chopped_hp_demod_diff = util.delete_picked(x_chopped_hp_demod_diff, 16, 15)
    t_diff, x_chopped_hp_demod_lp_diff = util.difference(x_chopped_hp_demod_lp, t)

    # low pass
    f_diff, df, X_chopped_hp_demod_diff = util.t_to_f(x_chopped_hp_demod_diff, dt, retstep=True)
    lp = util.low_pass_filter(f_diff, 10 * 1e3)
    X_chopped_hp_demod_diff_lp = X_chopped_hp_demod_diff * lp
    _, _, x_chopped_hp_demod_diff_lp = util.f_to_t(X_chopped_hp_demod_diff_lp, df, 0, retstep=True)

    # sampled
    t_diff_diff_lp_smp, x_chopped_hp_demod_diff_lp_smp = util.sample(x_chopped_hp_demod_diff_lp, t_diff, 32, 7)
    t_diff_lp_diff_smp, x_chopped_hp_demod_lp_diff_smp = util.sample(x_chopped_hp_demod_lp_diff, t_diff, 32, 0)

    ## Time domain plot
    ax_time_domain = None
    fig, ax_time_domain = util.plot(t, x, label='origin signal', ax=ax_time_domain, show=False, block=True)
    fig, ax_time_domain = util.plot(t, x+noise, label='origin signal with noise', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_chopped, label='chopped signal', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_smp, label='smp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_smp_chopped, label='smp chop', ax=ax_time_domain, show=False, block=True)
    fig, ax_time_domain = util.plot(t_cds, x_smp_chopped_cds, label='smp chop cds', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_chopped_hp, label='chop hp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_smp_chopped_hp, label='smp chop hp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_chopped_hp_demod, label='chop hp demod', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, x_chopped_hp_demod_lp, label='chop hp demod lp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff, x_chopped_hp_demod_diff, label='chopped hp demod diff', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff, x_chopped_hp_demod_diff_lp, label='chopped hp demod diff lp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff_diff_lp_smp, x_chopped_hp_demod_diff_lp_smp, label='chop hp demod diff lp smp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff, x_chopped_hp_demod_lp_diff, label='chop hp demod lp diff', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff_diff_lp_smp, x_chopped_hp_demod_lp_diff_smp, label='chop hp demod lp diff smp', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t_diff, util.delete_picked(x_chopped_hp_demod_diff, 16, 15), label='chopped signal hp demod diff', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, np.real(x_chopped_hp_diff), label='chopped signal high pass real', ax=ax_time_domain, show=False, block=True)
    # fig, ax_time_domain = util.plot(t, np.imag(x_chopped_hp_diff), label='chopped signal high pass imag', ax=ax_time_domain, show=False, block=True)

    ## Frequency domain plot
    ax_freq_domain = None
    # fig, ax_freq_domain = util.plot(f, np.real(X), label='original signal real', ax=ax_freq_domain, show=False, block=True)
    # fig, ax_freq_domain = util.plot(f, np.imag(X), label='original signal imag', ax=ax_freq_domain, show=False, block=True)
    # fig, ax_freq_domain = util.plot(f, np.real(X_chopped), label='chopped signal real', ax=ax_freq_domain, show=False, block=True)
    # fig, ax_freq_domain = util.plot(f, np.imag(X_chopped), label='chopped signal imag', ax=ax_freq_domain, show=False, block=True)
    # fig, ax_freq_domain = util.plot(f, hp, label='high pass filter', ax=ax_freq_domain, show=False, block=True)
    # fig, ax_freq_domain = util.plot(f, np.abs(hp), label='high pass filter', ax=ax_freq_domain, show=False, block=True)

    ## Bolt diagram
    ax_power, ax_phase = None, None
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_chopped + 1e-2, label='chopped signal', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_smp_chopped + 1e-6, label='smp chop', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    fig, ax_power, ax_phase = util.plot_freq_domain(
        f_cds, X_smp_chopped_cds + 1e-6, label='smp chop cds', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_smp_chopped_hp + 1e-6, label='smp chop hp', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X + 1e-2, label='original signal', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, hp, label='high pass filter', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_chopped_hp + 1e-6, label='chopped high pass', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_chopped_hp_diff + 1e-6, label='chop high pass - chop', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_chopped_hp_demod + 1e-6, label='chopped high pass demod', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f, X_chopped_hp_demod_lp + 1e-6, label='chopped hp demod lp', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f_diff, X_chopped_hp_demod_diff, label='chop hp demod diff', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
    # fig, ax_power, ax_phase = util.plot_freq_domain(
    #     f_diff, X_chopped_hp_demod_diff_lp+1e-6, label='chop hp demod diff lp', ax_power=ax_power, ax_phase=ax_phase, show=False, block= False)
      
 




    ## Show plot
    plt.show(block=False)
    
    ## close all
    input("Enter to exit...")
    plt.close('all')
