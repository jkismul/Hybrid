# import elephant
import neo
from elephant import spike_train_generation as stg
import pickle
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
import statsmodels.distributions as smd
import seaborn as sns
import scipy.stats as st

fr = pickle.load(open('data/firing_inhibition_on.p','rb'))
# print(np.shape(fr))
# print(fr)
frf = pickle.load(open('data/filtered_firing_inhibition_on.p','rb'))
# print(np.shape(fr))
# print(fr)
fs = [fr,frf]
num_tsteps = len(fr[0])
dt = 20./num_tsteps
t = np.arange(num_tsteps)*dt
for i,nam in enumerate(['filter_off','filter_on']):
    fr_ex = neo.AnalogSignal(100000*fs[i][0],units=pq.Hz,sampling_rate=1./dt*1*pq.Hz)
    fr_in = neo.AnalogSignal(100000*fs[i][1],units=pq.Hz,sampling_rate=1./dt*1*pq.Hz)
    spikes = stg.inhomogeneous_poisson_process(fr_ex,as_array=True)
    spikes_in = stg.inhomogeneous_poisson_process(fr_in,as_array=True)
    # # print(spikes)
    # plt.figure()
    # # plt.plot(spikes)
    # plt.plot(t,0.11*np.asarray(fr_ex),'r')
    # plt.hist(spikes,bins=np.linspace(0,21,200))
    # plt.show()

    # plt.figure()
    # # plt.plot(spikes)
    # plt.plot(t,0.11*np.asarray(fr_in),'r')
    # plt.hist(spikes_in,bins=np.linspace(0,21,200))
    # plt.show()

    pickle.dump(spikes,open('data/firing_signal_basal_inhibition_on_{}.p'.format(nam),'wb'))
    pickle.dump(spikes_in,open('data/firing_signal_apical_inhibition_on_{}.p'.format(nam),'wb'))
# pickle.dump(spikes,open('data/filt_firning_signal.p','wb'))
# pickle.dump(spikes_in,open('data/filt_firning_signal_in.p','wb'))

# fr_inh = neo.AnalogSignal(fr[1],units="1/ms",sampling_rate=1./dt*1000*pq.Hz)
#
# print(fr_ex)
# ex = stg.inhomogeneous_poisson_process(fr_ex)
# print(50*pq.Hz)
# spikes = stg.homogeneous_poisson_process(50*pq.Hz, t_start=0*pq.ms, t_stop=1000*pq.ms)
# spikes = stg.homogeneous_poisson_process(fr_ex, t_start=0*pq.ms, t_stop=1000*pq.ms)
#
# print(spikes)

# print(ex)
# plt.figure()
# plt.plot(fr_ex)
# plt.show()
# inh = stg.inhomogeneous_poisson_process(fr[1])

# ai = np.linspace(0,20,7)
# # ai=np.arange(0,20,)
# # print(ai.astype('int'))
# num_tsteps = 100
# num_elecs = 1
# dt = 0.1
# z = np.arange(num_elecs) * 100
# t = np.arange(num_tsteps) * dt
#
# coords = np.array([z]).T * pq.um
#
# laminar_LFP = np.random.normal(size=(num_elecs, num_tsteps))
# # print(laminar_LFP)
# pos_neo = z * pq.um
# lfp_neo = neo.AnalogSignal(laminar_LFP, units="uV", sampling_rate=1 / dt * 1000 * pq.Hz)
# # csd_estimate = elephant.current_source_density.estimate_csd(lfp_neo.T, coords, diam=100 * pq.um,  # h=25*pq.um,
#











# class my_pdf(st.rv_continuous):
#     def _pdf(self,x):
#         # return 2
#         # return fr[0]
#         return 3*x**2
#
# my_cv = my_pdf(a=0,b=20,name='my_pdf')
# draw = my_cv.rvs(size=2)
# # print(my_cv.pdf(t))
# z=9
# plt.figure()
# plt.scatter(np.arange(0,z),(my_cv.rvs(size=z)))
# # plt.plot(my_cv.pdf(t))
# plt.show()

#
# print(my_cv.rvs(1))
# plt.figure()
# plt.plot(t,my_cv.rvs(t))
# plt.show()
# plt.figure()
# plt.plot(fr[0])
# plt.show()
# plt.figure()
# sns.kdeplot(fr[0])
# # plt.plot(t,fr[0],'r--')
# plt.show()
# ecdf = smd.EPDF(fr[0])
# plt.figure()
# plt.plot(ecdf.x,ecdf.y)
# plt.show()
