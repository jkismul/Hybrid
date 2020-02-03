import sys
sys.path.insert(0, '/dependencies/')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from dependencies import plotting_convention
from scipy.optimize import minimize
from scipy.stats import norm, skewnorm
import pandas
import pickle


if len(sys.argv) == 2:
    mode = sys.argv[1]
else:
    print('No mode chosen. Defaulting to blocked.')
    mode = 'blocked'

kerns = pickle.load(open('data/kernels.p','rb'))
dt_k = pickle.load(open('data/dts.p','rb'))

if mode == 'on':
    # kerns_to_run = [0,1,3,4]
    kerns_to_run = [0,1]

    df = pandas.read_csv('data/inhib_on.csv')
    data = np.asarray(df['inhib_on'])
else:
    kerns_to_run = [0,1]
    df = pandas.read_csv('data/inhib_blocked.csv')
    data = np.asarray(df['inhib_blocked'])


param_idx = []
for i in kerns_to_run:
    param_idx.append(i*3)
    param_idx.append(i * 3+1)
    param_idx.append(i * 3+2)

# parameters
num_tsteps = len(df['x'])
kernel_num_tsteps = len(kerns[0])
num_kernels = len(kerns[1])
dt = 20./num_tsteps

dt_k = dt_k/kernel_num_tsteps
t = np.arange(num_tsteps) * dt

kernels = []

for idx in kerns_to_run:
    k_ = kerns[1][idx]
    kernels.append(k_)
num_kernels_to_run = len(kernels)

num_params_per_kernel = 3
num_params_per_kernel = 4

# initial conditions affect convergence,
# so reroll inits to find one that converges to a possible global minima
def reroll():
    x0 = [np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5),
          np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5),
          np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5),
          np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5),
          np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5),
          np.random.randint(0, 20), np.random.randint(0, 10), np.random.randint(0, 5), np.random.uniform(low=2, high=5)
          ]
    return x0

bounds = [
    #   loc           std       amp
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10],
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10],
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10],
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10],
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10],
    [t[0], t[-1]], [1e-7, 10], [0, 5],[0,10]
    ]

def minimize_firing_rates(x, *args):
    data = args[0]
    fit_ = np.zeros(len(data[0]))
    for idx in range(num_kernels_to_run):
        # RV = norm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel])
        RV = skewnorm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel],a=x[3 + idx * num_params_per_kernel])
        amp_ = x[2 + idx * num_params_per_kernel]
        firing_rate_ = amp_ * RV.pdf(t)[::-1]
        fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same",method='direct')
    return np.sum((data - fit_)**2)


args = [data]
err = 100
teller = 0
while err > 1.6:
    teller +=1
    x0 = reroll()
    res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, tol=1e-25, options={'eps': 1e-4})
    if minimize_firing_rates(res['x'],args) < err:
        err = minimize_firing_rates(res['x'],args)
    print('iteration:', teller, ', error:',minimize_firing_rates(res['x'],args))

print('predicted {}'.format(res['x'][param_idx].round(1)))
print('initial  ',np.asarray(x0)[param_idx])
print('error pred',minimize_firing_rates(res['x'],args))
print(res['message'])

pickle.dump(res['x'][param_idx],open('data/pdf_params.p','wb'))


fit = np.zeros(num_tsteps)
firing_rates_opt = []
for idx in range(num_kernels_to_run):
    # RV = norm(loc=res.x[0 + idx * num_params_per_kernel],scale=res.x[1 + idx * num_params_per_kernel])
    RV = skewnorm(loc=res.x[0 + idx * num_params_per_kernel], scale=res.x[1 + idx * num_params_per_kernel],
               a=res.x[3 + idx * num_params_per_kernel])
    amp_ = res.x[2 + idx * num_params_per_kernel]
    firing_rate_ = amp_ * RV.pdf(t)[::-1]
    firing_rates_opt.append(firing_rate_)
    fit += ss.convolve(firing_rate_, kernels[idx], mode="same",method='direct')

fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]",ylabel='Voltage [mV]')
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
coll = ['r','b']
lines = []
line_names = []
for idx in range(num_kernels_to_run):
    l_, = ax_fr.plot(t, firing_rates_opt[idx],c=coll[idx])
    ax_k.plot(np.arange(kernel_num_tsteps) * dt_k,kernels[idx],c=coll[idx])

    lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(t, data, c='k')
ax_sig.plot(t, fit, c='gray', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_play_num_kernels_{}.png".format(num_kernels))








fit = np.zeros(num_tsteps)
firing_rates_opt = []
for idx in range(num_kernels_to_run):
    RV = norm(loc=res.x[0 + idx * num_params_per_kernel],scale=res.x[1 + idx * num_params_per_kernel])
    amp_ = res.x[2 + idx * num_params_per_kernel]
    long = RV.pdf(t)
    maxx = np.max(long)
    maxx_arg = np.argmax(long)
    if idx == 1:
        diff = 5
        long[maxx_arg+diff:]= long[maxx_arg:-diff]
        long[maxx_arg:maxx_arg+diff]=maxx

        # long[maxx_arg+diff:]= long[maxx_arg:-diff]
        # long[maxx_arg:maxx_arg+diff]=np.linspace(maxx,maxx*0.9,diff)


    firing_rate_ = amp_ * long

    firing_rates_opt.append(firing_rate_)
    fit += ss.convolve(firing_rate_, kernels[idx], mode="same",method='direct')

fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]",ylabel='Voltage [mV]')
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
coll = ['r','b']
lines = []
line_names = []
for idx in range(num_kernels_to_run):
    l_, = ax_fr.plot(t, firing_rates_opt[idx],c=coll[idx])
    ax_k.plot(np.arange(kernel_num_tsteps) * dt_k,kernels[idx],c=coll[idx])

    lines.append(l_)
    line_names.append("firing rate fit {}".format(idx))
#######
df = pandas.read_csv('data/inhib_blocked.csv')
num_tsteps = len(df['x'])
kernel_num_tsteps = len(kerns[0])
num_kernels = len(kerns[1])
dt = 20./num_tsteps

dt_k = dt_k/kernel_num_tsteps
t2 = np.arange(num_tsteps) * dt
data = df['inhib_blocked']
#######
ax_sig.plot(t2, data, c='k')
ax_sig.plot(t, fit, c='gray', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_play_num_kernels_{}_blk.png".format(num_kernels))







###########################################33

#
#
# def minimize_firing_rates2(x, *args):
#     x0 = pickle.load(open('data/pdf_params.p', 'rb'))
#     # print(x0)
#     data = args[0]
#     fit_ = np.zeros(len(data[0]))
#     for idx in range(num_kernels_to_run):
#         # RV = norm(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel])
#         RV = skewnorm(loc=x0[0], scale=x0[1],a=x[1])
#         # RV = alpha(loc=x[0 + idx * num_params_per_kernel], scale=x[1 + idx * num_params_per_kernel],)
#
#         amp_ = x[0]#x0[3]
#         firing_rate_ = amp_ * RV.pdf(t2)
#         fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same",method='direct')
#     return np.sum((data - fit_)**2)
#
# df2 = pandas.read_csv('data/inhib_blocked.csv')
# num_tsteps2 = len(df2['x'])
#
# dt2 = 20./num_tsteps2
#
# t2 = np.arange(num_tsteps2) * dt2
# data2 = np.asarray(df2['inhib_blocked'])
# args2 = [data2]
# # xx = np.random.uniform(low=2, high=5)
# xx = [np.random.randint(0,5), np.random.uniform(low=2, high=5)]
#
# bounds2 = [[0,5],[0,100]]
# err = 500
# telle = 0
# # while err > 5:
# #     telle +=1
# #     xx = [np.random.randint(0, 5), np.random.uniform(low=2, high=5)]
# #     res2 = minimize(minimize_firing_rates2, xx, args=args2, bounds=bounds2, tol=1e-25, options={'eps': 1e-4})
# #     print(telle,minimize_firing_rates2(res2['x'], args2))
# # print(minimize_firing_rates2(res2['x'], args2))
#
#
# # print(res.x[1],res2.x)
#
# fit2 = np.zeros(num_tsteps2)
# firing_rates_opt = []
# for idx in range(num_kernels_to_run):
#     A = res.x[3+idx*num_params_per_kernel]
#     aa = res.x[2 + idx * num_params_per_kernel]
#     if idx == 1:
#         A = 100#res2.x[1]
#         aa = 10#res2.x[0]
#     # RV = norm(loc=res.x[0 + idx * num_params_per_kernel],scale=res.x[1 + idx * num_params_per_kernel])
#     RV = skewnorm(loc=res.x[0 + idx * num_params_per_kernel], scale=res.x[1 + idx * num_params_per_kernel],
#                a=A)# * num_params_per_kernel])
#      # a = res.x[3 + idx * num_params_per_kernel])
#     # amp_ = res.x[2 + idx * num_params_per_kernel]
#     amp_ = aa#res2.x[0]
#
#     firing_rate_ = amp_ * RV.pdf(t2)
#     firing_rates_opt.append(firing_rate_)
#     fit2 += ss.convolve(firing_rate_, kernels[idx], mode="same",method='direct')
#
#
# fig = plt.figure(figsize=[9, 4])
# fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
# ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]",ylabel='Voltage [mV]')
# ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
# ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
#
# lines = []
# line_names = []
# for idx in range(num_kernels_to_run):
#     l_, = ax_fr.plot(t2, firing_rates_opt[idx])
#     ax_k.plot(np.arange(kernel_num_tsteps) * dt_k,kernels[idx])
#
#     lines.append(l_)
#     line_names.append("firing rate fit {}".format(idx))
#
# # ax_sig.plot(t2, data2, c='k')
# ax_sig.plot(t, data, c='red', ls='--')
#
# ax_sig.plot(t2, fit2, c='gray', ls='--')
#
# fig.legend(lines, line_names, frameon=False, ncol=4)
# plotting_convention.simplify_axes(fig.axes)
# plt.savefig("plots/aa_convolve_play_num_kernels_{}.png".format(num_kernels))