'''
Simple plot.
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

def plot(
    x_data: np.ndarray,  # 自变量数据
    y_data: np.ndarray,  # 因变量数据
    label: str = 'Data',
    ax=None,  # 可选的Axes对象
    show=False,
    block=False,
):
    '''
    Simple plot.
    '''
    # 检查是否提供了Axes对象，如果没有则创建新的Figure和Axes
    if ax is None:
        fig, ax = plt.subplots()
        plt.grid(True)
    else:
        fig = ax.figure  # 获取Axes所属的Figure对象

    ax.plot(x_data, y_data, label=label)  # 使用提供的x_data和y_data进行作图
    # ax.set_title('Data Plot')
    # ax.set_xlabel('X-Axis')
    # ax.set_ylabel('Y-Axis')
    ax.legend(loc='upper right')

    plt.tight_layout()
    if show:
        plt.show(block=block)

    return fig, ax  # 返回Figure和Axes对象

def plot_freq_domain(
    f: np.ndarray,
    X: np.ndarray,
    label: str = 'Data',
    ax_power = None,
    ax_phase = None,
    show=False,
    block=False,
):
    if ax_power is None or ax_phase is None:
        fig, (ax_power, ax_phase) = plt.subplots(2,1, sharex=True)
        ax_power.grid(True)
        ax_phase.grid(True)
    else:
        fig = ax_power.figure
    
    _, ax_power = plot(f, np.abs(X), label=label, ax=ax_power, show=show, block=block)
    _, ax_phase = plot(f, np.angle(X, deg=True), label=label, ax=ax_phase, show=show, block=block)

    return fig, ax_power, ax_phase
 
