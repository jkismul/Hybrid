import sys
sys.path.insert(0, '/dependencies/')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from dependencies import plotting_convention
from scipy.optimize import minimize
from scipy.stats import norm
import pandas
import pickle

kerns = pickle.load(open('data/kernels.p','rb'))
dt_k = pickle.load(open('data/dts.p','rb'))
kerns_to_run=[0,1]
# kerns_to_run = [0,1,3,4]
# kerns_to_run = [0,1,2,3,4,5]
param_idx = []
for i in kerns_to_run:
    param_idx.append(i*3)
    param_idx.append(i * 3+1)
    param_idx.append(i * 3+2)

#3read real data
df_on = pandas.read_csv('data/inhib_on.csv')
df_off = pandas.read_csv('data/inhib_blocked.csv')

# parameters
num_tsteps = len(df_off['x'])
kernel_num_tsteps = len(kerns[0])
num_kernels = len(kerns[1])
dt = 20./num_tsteps

dt_k = dt_k/kernel_num_tsteps
t = np.arange(num_tsteps) * dt

colors = ['b', 'r', 'orange']
colors_opt = ['cyan', 'pink', 'yellow']
kernels = []

for idx in kerns_to_run:
    k_ = kerns[1][idx]
    kernels.append(k_)
num_kernels_to_run = len(kernels)

num_params_per_kernel = 3

data = np.asarray(df_off['inhib_blocked'])
## VELDIG AVHENGIG AV IINiTALBETINGELSER
## LITT TESTING KAN TYDE PÅ AT HØYE VERDIER TREFFER BEDRE ENN LAVE
x0 = [15, 5, 1,
      15, 5, 1,
      15, 5, 1,
      15, 5, 1,
      15, 5, 1,
      15, 5, 1
      ]

bounds = [
    #   loc           std       amp
    [t[0], t[-1]], [1e-7, 10], [0, 5],
    [t[0], t[-1]], [1e-7, 10], [0, 5],
    [t[0], t[-1]], [1e-7, 10], [0, 5],
    [t[0], t[-1]], [1e-7, 10], [0, 5],
    [t[0], t[-1]], [1e-7, 10], [0, 5],
    [t[0], t[-1]], [1e-7, 10], [0, 5]
    ]

def minimize_firing_rates(x, *args):
    data = args[0]
    fit_ = np.zeros(len(data[0]))
    for idx in range(num_kernels_to_run):
        RV = norm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel])
        amp_ = x[2 + idx * num_params_per_kernel]
        firing_rate_ = amp_ * RV.pdf(t)
        fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same")
    return np.sum((data - fit_)**2)


args = [data]
res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds,tol=1e-25,options={'eps':1e-4})
print('predicted {}'.format(res['x'][param_idx].round(1)))
print('initial  ',np.asarray(x0)[param_idx])
print('error pred',minimize_firing_rates(res['x'],args))
print(res['message'])

fit = np.zeros(num_tsteps)
firing_rates_opt = []
for idx in range(num_kernels_to_run):
    RV = norm(loc=res.x[0 + idx * num_params_per_kernel],scale=res.x[1 + idx * num_params_per_kernel])
    amp_ = res.x[2 + idx * num_params_per_kernel]
    firing_rate_ = amp_ * RV.pdf(t)
    firing_rates_opt.append(firing_rate_)
    fit += ss.convolve(firing_rate_, kernels[idx], mode="same")

fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time")
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time")

lines = []
line_names = []
for idx in range(num_kernels_to_run):
    l_, = ax_fr.plot(t, firing_rates_opt[idx])#, c=colors_opt[idx], ls='--')
    ax_k.plot(np.arange(kernel_num_tsteps) * dt_k,kernels[idx])#, c=colors[idx])

    lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(t, data, c='k')
ax_sig.plot(t, fit, c='gray', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_play_num_kernels_{}.png".format(num_kernels))