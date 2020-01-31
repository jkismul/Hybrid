import LFPy
import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv)>1:
    iters = int(sys.argv[1])
else:
    iters = 1

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
        s.set_spike_times(np.array([20.]))


cell_parameters = {
    'morphology': morphology,
    'cm': 1.0,  # membrane capacitance
    'Ra': 150.,  # axial resistance
    'v_init': -65.,  # initial crossmembrane potential
    'passive': True,  # turn on NEURONs passive mechanism for all sections
    # 'passive': False,  # turn on NEURONs passive mechanism for all sections

    'passive_parameters': {'g_pas': 1. / 30000, 'e_pas': -65},
    'nsegs_method': 'lambda_f',  # spatial discretization method
    'lambda_f': 100.,  # frequency where length constants are computed
    'dt': 2. ** -3,  # simulation time step size
    'tstart': -300.,  # start time of simulation, recorders start at t=0
    'tstop': 100.,  # stop simulation at 100 ms.
}
# Define synapse parameters
synapse_parameters = {
    'idx':[],
    'e': 0,  # reversal potential
    # 'syntype': 'ExpSyn',  # synapse type
    # 'tau': 1.,  # synaptic time constant
    'syntype': 'Exp2Syn',
    'tau1':.2,
    'tau2':1.,

    'weight': .001,  # synaptic weight
    'record_current': True,  # record synapse current
}

# soma_pos = np.array([0.,0.,0.])
p = []
print("running simulation...")

n_syn=200
np.random.seed(10)
# for j in range(iters):
#     print('iteration', j+1, 'of', iters)
#     np.random.seed(j+10)
for i in range(6):
    # Create cell
    cell = LFPy.Cell(**cell_parameters)
    cell.set_rotation(x=4.99, y=-4.33, z=3.14)

    if i > 2:
        synapse_parameters['e'] = -80
    else:
        synapse_parameters['e'] = 0
    if i in [0, 3]:
        # np.random.seed(1234+j)
        np.random.seed(1234)
        insert_synapses(synapse_parameters,'dend',n_syn)
    if i in [1,4]:
        # np.random.seed(1234+j)
        np.random.seed(1234)
        insert_synapses(synapse_parameters,'apic',n_syn)
    else:
        np.random.seed(1234)
        insert_synapses(synapse_parameters,'allsec',n_syn)



    synapse = LFPy.Synapse(cell, **synapse_parameters)
    synapse.set_spike_times(np.array([20.]))
    # synapse.set_spike_times(np.array([10.]))

    cell.simulate(rec_imem=True,rec_vmem=True,rec_current_dipole_moment=True)

    P = cell.current_dipole_moment
    p.append(P)

print("done")

plt.figure()
p = np.asarray(p)/n_syn
legs = ['ed','ea','ef','id','ia','if']
for i in range(6):
    plt.plot(cell.tvec,p[i,:,2],label=legs[i])
plt.legend()
plt.savefig('plots/kernels.png')

# plt.plot()
# plt.plot(cell.tvec,cell.vmem[0])
# plt.show()