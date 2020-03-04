import LFPy
import numpy as np
import matplotlib.pyplot as plt
import pickle

morphology = 'morphologies/L5_Mainen96_LFPy.hoc'

def insert_synapses(synparams, section, n):
    if section == 'dend':
        maxim = -50
        minim=-1000
    if section == 'apic':
        maxim=1000
        minim=500
    if section=='allsec':
        maxim=1000
        minim=-1000
    '''find n compartments to insert synapses onto'''
    idx = cell.get_rand_idx_area_norm(section=section, nidx=n,z_min=minim,z_max=maxim)

    #Insert synapses in an iterative fashion
    for i in idx:
        synparams.update({'idx' : int(i)})

        # Create synapse(s) and setting times using the Synapse class in LFPy
        s = LFPy.Synapse(cell, **synparams)
        s.set_spike_times(np.array([cell.tstop/2.]))


cell_parameters = {
    'morphology': morphology,
    'cm': 1.0,  # membrane capacitance
    'Ra': 150.,  # axial resistance
    'v_init': -65.,  # initial crossmembrane potential
    # 'passive': True,  # turn on NEURONs passive mechanism for all sections
    'passive': False,  # turn on NEURONs passive mechanism for all sections
    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'nsegs_method': 'lambda_f',  # spatial discretization method
    'lambda_f': 100.,  # frequency where length constants are computed
    'dt': 2. ** -3,  # simulation time step size
    'tstart': -300.,  # start time of simulation, recorders start at t=0
    'tstop': 50.,  # stop simulation at 100 ms.
}
# Define synapse parameters
synapse_parameters = {
    'idx':[],
    'e': 0,  # reversal potential
    # 'syntype': 'ExpSyn',  # synapse type
    # 'tau': 0.005,  # synaptic time constant
    'syntype': 'ExpSynI',
    'tau1':1.,
    'tau2':3.,
    'tau':.1,
    'weight': .1001,  # synaptic weight
    'record_current': True,  # record synapse current
}

p = []
print("running simulation...")
n_syn=200
np.random.seed(10)
for i in range(6):
    # Create cell
    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33, z=3.14)

    if i > 2:
        synapse_parameters['e'] = -80
        synapse_parameters['weight']=-.1001
        # synapse_parameters['tau']=30.0
    else:
        synapse_parameters['e'] = 0
    if i in [0, 3]:
        np.random.seed(1234)
        insert_synapses(synapse_parameters,'dend',n_syn)
    if i in [1,4]:
        np.random.seed(1234)
        # synapse_parameters['tau']=3.
        insert_synapses(synapse_parameters,'apic',n_syn)
    else:
        np.random.seed(1234)
        insert_synapses(synapse_parameters,'allsec',n_syn)

    cell.simulate(rec_imem=True,rec_vmem=True,rec_current_dipole_moment=True)

    P = cell.current_dipole_moment
    p.append(P)
print("done")
print(np.shape(cell.imem),np.shape(cell.tvec))
#createfigure
plt.figure()
p = np.asarray(p)/n_syn
legs = ['ed','ea','ef','id','ia','if']
for i in range(6):
    plt.plot(cell.tvec,p[i,:,2],label=legs[i])
plt.legend()
plt.savefig('plots/kernels.png')

pickle.dump([cell.tvec,p[:,:,2]],open('data/kernels.p','wb'))
pickle.dump(cell_parameters['tstop'],open('data/dts.p','wb'))