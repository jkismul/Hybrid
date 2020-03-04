import sys
sys.path.insert(0, '/dependencies/')
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from dependencies import plotting_convention
from scipy.optimize import minimize
import pandas
import pickle

kerns = pickle.load(open('data/kernels.p','rb'))
dt_k = pickle.load(open('data/dts.p','rb'))

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

num_params_per_kernel = 4

num_draws = 1000
number = num_tsteps

hist_bins = np.linspace(0,20,number+1)

x0 = pickle.load(open('data/pdf_params.p','rb'))
bounds=[]

for i in range(int((len(hist_bins)/2))):#start up
    # bounds.append([0, 10])
    bounds.append([x0[i],x0[i]])
for i in range(int(len(hist_bins)-(len(hist_bins)/2))): #end up
    # print(i+int((len(hist_bins)/2)))
    bounds.append([0, 10])
    # bounds.append([x0[i+int((len(hist_bins)/2))],x0[i+int((len(hist_bins)/2))]])
# bounds.append([0,10]) #amp up
bounds.append([x0[len(hist_bins)+1],x0[len(hist_bins)+1]])
for i in range(int((len(hist_bins)/2))):#start up
    bounds.append([0, 10])
    # print(2+i+1*int((len(hist_bins)/2)))
    # bounds.append([x0[2+i+1*int((len(hist_bins)/2))],x0[2+i+1*int((len(hist_bins)/2))]])

for i in range(int(len(hist_bins)-(len(hist_bins)/2))): #end up
    bounds.append([0, 10])
    # print(2+i+3*int((len(hist_bins)/2)))
    # bounds.append([x0[2+i+2*int((len(hist_bins)/2))],x0[2+i+2*int((len(hist_bins)/2))]])

bounds.append([0,10]) #amp up

def reroll():
    x0 = pickle.load(open('data/pdf_params.p', 'rb'))
    first = x0[:72]
    second = np.zeros(len(x0)-len(first))
    second = x0[74:-3]
    third = x0[-1]
    return np.hstack((first,second,third))
    # x0 = x0[:-4] #difference in data file length
    # return x0

def minimize_firing_rates(x, *args):
    data = args[0]
    fit_ = np.zeros(len(data[0]))
    for idx in range(num_kernels_to_run):
        amp_=x[number*(idx+1)+idx]
        firing_rate_ = amp_*x[(idx)*number+idx:(idx+1)*number+idx]
        fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same",method='direct')
    return np.sum((data - fit_)**2)


args = [data]
err = 100
teller = 0
while err > .025:
    teller +=1
    if teller ==3:
        break
    x0 = reroll()
    # print(np.shape(x0))
    # print('---',len(x0), len(bounds))
    res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, tol=1e-25, options={'eps': 1e-4})
    if minimize_firing_rates(res['x'],args) < err:
        err = minimize_firing_rates(res['x'],args)
    print('iteration:', teller, ', error:',minimize_firing_rates(res['x'],args))

# print('predicted {}'.format(res['x'][param_idx].round(1)))
# print('initial  ',np.asarray(x0)[param_idx])
print('error pred',minimize_firing_rates(res['x'],args))
print(res['message'])

# pickle.dump(res['x'][param_idx],open('data/pdf_params.p','wb'))


##FILTER FIRING
dt = 20/74.
fs = 1/dt*1000
fc=200
w = fc/(fs*0.5)
b,a = ss.butter(2,w,'lowpass')


fit = np.zeros(num_tsteps)
firing_rates_opt = []
filtfire = []
fit_lp = np.zeros(num_tsteps)

fit_s = [np.zeros((num_tsteps)),np.zeros((num_tsteps))]

for idx in range(num_kernels_to_run):
    amp_ = res.x[number * (idx + 1) + idx]
    firing_rate_ = amp_ * res.x[(idx) * number + idx:(idx + 1) * number + idx]

    # firing_rate_ = res.x[-1]*res.x[:-1]
    firing_rates_opt.append(firing_rate_)
    filtr = ss.filtfilt(b,a,firing_rate_)
    filtfire.append(filtr)
    # firing_rates_opt.append(filtr)
    fit += ss.convolve(firing_rate_, kernels[idx], mode="same",method='direct')
    fit_lp += ss.convolve(filtfire[idx],kernels[idx],mode='same')#,method='direct')
    fit_s[idx] = ss.convolve(firing_rate_,kernels[idx],mode='same')

filtfire=np.where(np.asarray(filtfire)>0,np.asarray(filtfire),0)

pickle.dump(firing_rates_opt,open('data/firing_inhibition_off.p','wb'))
pickle.dump(filtfire,open('data/filtered_firing_inhibition_off.p','wb'))
pickle.dump(fit_s,open('data/fits_inhibition_off.p','wb'))

fig = plt.figure(figsize=[9, 4])
fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
ax_k = fig.add_subplot(131, title="kernels", xlabel="time [ms]",ylabel='Voltage [mV]')
ax_fr = fig.add_subplot(132, title="Firing rates", xlabel="time [ms]")
ax_sig = fig.add_subplot(133, title="signal (convolution)", xlabel="time [ms]",ylabel='Voltage [mV]')
coll = ['r','b']
coll2=['orange','cyan']
lines = []
line_names = []
for idx in range(num_kernels_to_run):
    l_, = ax_fr.plot(t, firing_rates_opt[idx],c=coll2[idx],alpha=0.5)
    l2_, = ax_fr.plot(t, filtfire[idx],c=coll[idx])
    ax_k.plot(np.arange(kernel_num_tsteps) * dt_k,kernels[idx],c=coll[idx])

    lines.append(l_)
    lines.append(l2_)
    line_names.append("firing rate fit {}".format(idx))

ax_sig.plot(t, data, c='k')
ax_sig.plot(t, fit, c='gray', ls='--')
ax_sig.plot(t,fit_lp, c='red', ls='--')
ax_sig.plot(t,fit_s[0], c='orange', ls='--')
ax_sig.plot(t,fit_s[1], c='green', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_play_num_kernels_{}_blocks.png".format(num_kernels))
