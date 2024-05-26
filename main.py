import numpy as np
import matplotlib.pyplot as plt

import util

def filter_gen(f, fz, fp):
    return (1 + (1j) * f / fz) / (1 + (1j) * f / fp)    

def ideal_lp_gen(f, f0):
    _ = np.ones(f.shape)
    for i in range(len(f)):
        if np.abs(f[i]) >= f0:
            _[i] = 0
    return _

def lp_gen(f, f0):
    return 1 / (1 + (1j) * f / f0)

if __name__ == '__main__':
    dt = 5e-7
    runtime = 2.
    length = int(runtime / dt)
    t, dt = np.linspace(0,runtime,length, retstep=True, endpoint=False)
    f0 = 50
    x = np.sin(2 * np.pi * f0 * t)

    x_diff = x[1:] - x[:-1]

    chop = np.zeros(t.shape)

    for _ in range(len(t)):
        if _ % 32 < 16:
            chop[_] = 1
        else:
            chop[_] = -1

    y = x * chop
    # y = y + 1e-3 * np.sin(2 * np.pi * 3e1 * t)

    f, y_f, y_power, y_phase = util.power_and_phase(y, dt, log=False)

    f_filter = filter_gen(f, 1e3, 10 * 1e3)
    # f_filter = filter_gen(f, 1e1, 1e2)
    # f_filter = np.ones(f.shape) * 10
    f_filter_ang = np.angle(f_filter)
    t_filter = np.fft.ifft(np.fft.ifftshift(f_filter))

    # f_filter = np.abs(f_filter)
    # f_filter = 1 - ideal_lp_gen(f, 1e3)

    # conv_chop = np.convolve(t_filter[0:10000], chop, mode='same')

    y_f_filtered = f_filter * y_f

    y_filtered_power = np.abs(y_f_filtered)


    y_filtered_power = y_filtered_power / y_filtered_power.max()
    y_power = y_power / y_power.max()

    z = np.fft.ifft(np.fft.ifftshift(y_f_filtered)) / 10
    z_re = np.real(z)
    z_im = np.imag(z)
    z_de = z * chop
    # z_de = z

    t_diff = []
    z_diff = []
    for _ in range(len(z_de) - 1):
        t_diff.append(t[_])
        if _ % 16 == 15 :
            val = z_diff[-1]
        else:
            val = z_de[_+1] - z_de[_]
        z_diff.append(val)
    z_diff = np.array(z_diff)
    f_diff, z_diff_f, z_diff_power, z_diff_phase = util.power_and_phase(z_diff, dt, False)
    z_diff_power /= z_diff_power.max()

    z_diff_f_lp = z_diff_f * lp_gen(f_diff, 10 * 1e3) * (1 - ideal_lp_gen(f_diff, 1e3))
    z_diff_lp = np.fft.ifft(np.fft.ifftshift(z_diff_f_lp))

    z_diff_lp_sp, t_diff_sp = util.sample(z_diff_lp, t_diff, 32, 0)

    t_diff_sampled = []
    z_diff_sampled = []
    for _ in range(len(t_diff)):
        if _ % 32 == 0:
            t_diff_sampled.append(t_diff[_])
            z_diff_sampled.append(z_diff[_])
    z_diff_sampled = np.array(z_diff_sampled)

    # f_lp = ideal_lp_gen(f, 10*1e3)
    f_lp = lp_gen(f, 1*1e3)
    f_lp = np.abs(f_lp)
    z_f_lp = f_lp * np.fft.fftshift(np.fft.fft(z_de))


    z_lp = np.fft.ifft(np.fft.ifftshift(z_f_lp))

    t_chop_pos = []
    y_chop_pos = []
    for _ in range(len(t)):
        if _ % 32 == 0:
            t_chop_pos.append(t[_])
            y_chop_pos.append(z[_])

    y_chop_pos = np.array(y_chop_pos)

    t_chop_diff = t_chop_pos[:-1]
    y_chop_diff = y_chop_pos[1:] - y_chop_pos[:-1]

    t_chop_neg = []
    y_chop_neg = []
    for _ in range(len(t)):
        if _ % 32 == 16 + (-15):
            t_chop_neg.append(t[_])
            y_chop_neg.append(z[_])

    y_chop_neg = np.array(y_chop_neg)

    y_chop = y_chop_pos - y_chop_neg
    
    # f_chop, y_f_chop, 

    
    
    # t_cds = []
    # y_cds = []
    # for _ in range(len(y_chop)-16):
    #     y_cds.append(y_chop[_] - y_chop[_+1])
    #     t_cds.append(t_chop[_])
    

    _, z_f, z_power, z_phase = util.power_and_phase(z_de, dt, log=False)
    z_power = z_power / z_power.max()
    


    plt.figure()
    # plt.plot(t, x, label='sin')
    # plt.plot(t_diff, x_diff, label='sin diff')
    
    # plt.plot(t, y, label='chop')
    # plt.plot(t, z, label='filtered')
    # plt.plot(t, z_re, label='re')
    # plt.plot(t, z_im, label='im')
    # plt.plot(t_chop_pos, y_chop_pos, label='sampled')
    # plt.plot(t_chop_neg, y_chop_neg, label='picked 16')

    # plt.plot(t_diff, z_diff, label='de- difference')
    # plt.plot(t_diff, z_diff_lp, label='de- difference lp')
    plt.plot(t_diff_sp, z_diff_lp_sp, label='lp and sampled')
    # plt.plot(t_diff_sampled, z_diff_sampled, label='sampled difference')
    # plt.plot(t_chop_diff, y_chop_diff, label='difference of sampled')

    # plt.plot(t, z, label='out')
    # plt.plot(t, z_de, label='de')

    # plt.plot(t, t_filter, label='filter')
    # plt.plot(t, conv_chop, label='conv chop')
    # plt.plot(t, z_lp, label='lp')

    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show(block=False)

    plt.figure()
    # plt.plot(f, x_power, label='sin')
    # plt.plot(f, y_power, label='chop')
    # plt.plot(f, y_filtered_power, label='out')
    # plt.plot(f, z_power, label='de-')
    # plt.plot(f, z_f_lp, label='lp')
    plt.plot(f_diff, z_diff_power, label='diff')
    # plt.plot(f, np.abs(f_filter), label='filter')
    plt.grid(True)
    plt.legend(loc='upper right')
    # plt.show(block=False)

    # plt.figure()
    # plt.plot(f, f_filter_ang, label='filter')
    # plt.plot(f, np.unwrap(np.angle(y_f)), label='chop')
    # plt.plot(f, np.unwrap(np.angle(y_f_filtered)), label='filtered')
    # plt.grid(True)
    # plt.legend(loc='upper right')
    plt.show(block=True)


    
