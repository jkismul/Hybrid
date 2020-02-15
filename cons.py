import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from scipy.optimize import minimize
import pandas
import pickle

sys.path.insert(0, '/dependencies/')
from dependencies import plotting_convention


kerns = pickle.load(open('data/kernels.p','rb'))
dt_k = pickle.load(open('data/dts.p','rb'))

kerns_to_run = [0,1]

df = pandas.read_csv('data/inhib_on.csv')
data = np.asarray(df['inhib_on'])
tail = np.ones(len(data))*0#data[-1]

# data = np.hstack((data,tail))
# param_idx = []

# parameters
num_tsteps = len(df['x'])
kernel_num_tsteps = len(kerns[0])

num_kernels=len(kerns_to_run)
dt = 20./num_tsteps
dt_k = dt_k/kernel_num_tsteps
t = np.arange(num_tsteps) * dt
t = np.arange(len(data))*dt
t_k = np.arange(kernel_num_tsteps)*dt_k
kernels = []

for idx in kerns_to_run:
    k_ = kerns[1][idx]#[int(kernel_num_tsteps/2):]
    kernels.append(k_)
num_kernels_to_run = len(kernels)

# number = int(num_tsteps)
number=len(data)

hist_bins = np.linspace(0,20,number+1)

bounds=[]
silent = 0#80

# for i in range(int(len(hist_bins))-silent):#start up
#     bounds.append([0, 10])
for i in range(int((len(hist_bins)-1))):#start up
    bounds.append([0, 10])
# for i in range(int(len(hist_bins)-(len(hist_bins)/2)-silent)): #end up
#     bounds.append([0, 10])
# for i in range(silent-1): #end up
#     print('g')
#     bounds.append([0, 10])

bounds.append([0,10]) #amp up
# for i in range(int(len(hist_bins))-silent):#start up
#     bounds.append([0, 10])
for i in range(int((len(hist_bins)-1))):#start up
    bounds.append([0, 10])
# for i in range(int(len(hist_bins)-(len(hist_bins)/2)-silent)): #end up
#     bounds.append([0, 10])
# for i in range(silent-1): #end up
#     bounds.append([0, 10])
# bounds[-10:]=[0,0]
bounds.append([0,10]) #amp up
# initial conditions affect convergence,
# so reroll inits to find one that converges to a possible global minima
def reroll():
    x0=np.zeros(number)
    x1=np.zeros(number)
    x0 = np.append(x0,np.random.randint(0,5))
    x1 = np.append(x1,np.random.randint(0,5))
    x0 = np.append(x0,x1)
    return x0

def minimize_firing_rates(x, *args):
    data = args[0]
    fit_ = np.zeros(len(data[0]))
    for idx in range(num_kernels_to_run):
        amp_=x[number*(idx+1)+idx]
        firing_rate_ = amp_*x[(idx)*number+idx:(idx+1)*number+idx]
        fit_ += ss.convolve(firing_rate_, kernels[idx],mode="same")#,method='direct')

    return np.sum((data - fit_)**2)


args = [data]
err = 100
teller = 0
# while err > .019:
while err > .12:

    teller +=1
    if teller ==10:
        break
    x0 = reroll()
    print(np.shape(x0),np.shape(bounds))
    res = minimize(minimize_firing_rates, x0, args=args, bounds=bounds, tol=1e-25, options={'eps': 1e-4})
    if minimize_firing_rates(res['x'],args) < err:
        err = minimize_firing_rates(res['x'],args)
    print('iteration:', teller, ', error:',minimize_firing_rates(res['x'],args))

print('final error pred',minimize_firing_rates(res['x'],args))
print(res['message'])
pickle.dump(res['x'],open('data/pdf_params.p','wb'))

##FILTER FIRING
dt = 20/74.
fs = 1/dt*1000
fc=200
w = fc/(fs*0.5)
b,a = ss.butter(2,w,'lowpass')


fit = np.zeros(number)

# fit=np.zeros(kernel_num_tsteps)
firing_rates_opt = []
filtfire = []
fit_lp = np.zeros(number)

fit_s = [np.zeros(number),np.zeros(number)]

# fit_s = [np.zeros((kernel_num_tsteps)),np.zeros((kernel_num_tsteps))]

for idx in range(num_kernels_to_run):
    amp_ = res.x[number * (idx + 1) + idx]
    firing_rate_ = amp_ * res.x[(idx) * number + idx:(idx + 1) * number + idx]
    firing_rates_opt.append(firing_rate_)
    filtr = ss.filtfilt(b,a,firing_rate_)
    filtfire.append(filtr)
    fit += ss.convolve(firing_rate_, kernels[idx], mode="same")#,method='direct')
    fit_lp += ss.convolve(filtfire[idx],kernels[idx],mode='same')#,method='direct')
    fit_s[idx] = ss.convolve(firing_rate_,kernels[idx],mode='same')

filtfire=np.where(np.asarray(filtfire)>0,np.asarray(filtfire),0)

pickle.dump(firing_rates_opt,open('data/firing_inhibition_on.p','wb'))
pickle.dump(filtfire,open('data/filtered_firing_inhibition_on.p','wb'))
pickle.dump(fit_s,open('data/fits_inhib_on.p','wb'))

#PLOT
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

ax_sig.plot(t,data, c='k')
ax_sig.plot(t,fit, c='gray', ls='--')
ax_sig.plot(t,fit_lp, c='red', ls='--')
ax_sig.plot(t,fit_s[0], c='orange', ls='--')
ax_sig.plot(t,fit_s[1], c='green', ls='--')

fig.legend(lines, line_names, frameon=False, ncol=4)
plotting_convention.simplify_axes(fig.axes)
plt.savefig("plots/convolve_bn.png")