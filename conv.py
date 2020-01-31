import sys
sys.path.insert(0, '/dependencies/')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from dependencies import plotting_convention
from scipy.optimize import minimize
from scipy.stats import norm
import pandas


#create normally distributed firing rates per kernel
rv = norm(loc=25, scale=8)
rv1 = norm(loc=60,scale=10)
rv2 = norm(loc=80,scale=10)
rvs=[rv,rv1,rv2]

#3read real data
df_on = pandas.read_csv('data/inhib_on.csv')
df_off = pandas.read_csv('data/inhib_blocked.csv')

# parameters
num_tsteps = 1000
# num_tsteps = len(df_on['x'])
kernel_num_tsteps = 200
num_kernels = 3
dt = 0.1
t = np.arange(num_tsteps) * dt

taus = [0.5, 2, 3] #timeconstants
amps = [1, -0.7, -0.25] #amplitudes
colors = ['b', 'r', 'orange']
colors_opt = ['cyan', 'pink', 'yellow']
kernels = []
for idx in range(num_kernels):
    k_ = np.zeros(kernel_num_tsteps)
    k_[int(kernel_num_tsteps / 2):] = amps[idx] * np.exp(-t[:int(kernel_num_tsteps/2)] / taus[idx])
    kernels.append(k_)

t_locs = [25, 60, 80]
stds = [8, 10, 10]
amps = [1.8, 1, 0.7]

num_params_per_kernel = 3

correct = np.zeros(num_kernels*num_params_per_kernel)
for j,i in enumerate(range(0,num_kernels*num_params_per_kernel,3)):
    correct[i]= t_locs[j]
    correct[i+1]=stds[j]
    correct[i+2]=amps[j]

firing_rates = [amps[idx] * rvs[idx].pdf(t)
                for idx in range(num_kernels)]

data = np.zeros(num_tsteps)
for idx in range(num_kernels):
    data += ss.convolve(firing_rates[idx], kernels[idx], mode="same")

x0 = [50, 10, 1,
      50, 10, 1,
      50, 10, 1,
      ]
## VELDIG AVHENGIG AV IINiTALBETINGELSER
## LITT TESTING KAN TYDE PÅ AT HØYE VERDIER TREFFER BEDRE ENN LAVE
x0 = [50, 10, 1,
      90, 10, 1,
      90, 10, 1,
      ]

# bounds = [
#     [t[0], t[-1]], [0, 50], [0, 2],
#     [t[0], t[-1]], [0, 50], [0, 2],
#     [t[0], t[-1]], [0, 50], [0, 2],
#     ]
bounds = [
    #   loc           std       amp
    [t[0], t[-1]], [1e-7, 50], [0, 2],
    [t[0], t[-1]], [1e-7, 50], [0, 2],
    [t[0], t[-1]], [1e-7, 50], [0, 2],
    ]
# print(x0)
# print(bounds)
def blah(data,l1,l2,l3,s1,s2,s3,a1,a2,a3):
    RV1 = norm(loc=l1,scale=s1)
    RV2 = norm(loc=l2, scale=s2)
    RV3 = norm(loc=l3, scale=s3)
    fits = ss.convolve(a1*RV1.pdf(t),kernels[0],mode='same')+ss.convolve(a2*RV2.pdf(t),kernels[1],mode='same')+ ss.convolve(a3*RV3.pdf(t),kernels[2],mode='same')
    return np.sum((data-fits)**2)

def minimize_firing_rates(x, *args):
    data = args[0]
    fit_ = np.zeros(num_tsteps)
    for idx in range(num_kernels):
        RV = norm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel])
        # if scale = 0, the above whines. Adding a small number to scale
        # RV = norm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel]+1e-7)
        amp_ = x[2 + idx * num_params_per_kernel]
        firing_rate_ = amp_ * RV.pdf(t)
        fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same")
    return np.sum((data - fit_)**2)


args = [data]
res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds,tol=1e-25,options={'eps':1e-4})
print('predicted {}'.format(res['x'].round(1)))
print('correct  ',correct.round(1))
print('error pred',minimize_firing_rates(res['x'],args))
print('error correct',minimize_firing_rates(correct,args))
print(res['message'])

# print(res)
fit = np.zeros(num_tsteps)
firing_rates_opt = []
for idx in range(num_kernels):
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
for idx in range(num_kernels):
    l, = ax_fr.plot(t, firing_rates[idx], c=colors[idx])
    l_, = ax_fr.plot(t, firing_rates_opt[idx], c=colors_opt[idx], ls='--')
    ax_k.plot(np.arange(kernel_num_tsteps) * dt, kernels[idx][:kernel_num_tsteps], c=colors[idx])
    lines.append(l)
    line_names.append("Kernel {}".format(idx))
    lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(t, data, c='k')
ax_sig.plot(t, fit, c='gray', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_play_num_kernels_{}.png".format(num_kernels))