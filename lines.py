import matplotlib.pyplot as plt
import pandas
import numpy as np
import pickle

#read data
df_on = pandas.read_csv('data/inhib_on.csv')
df_off = pandas.read_csv('data/inhib_blocked.csv')

# "equalize" data
x_blocked = df_off['x']
y_blocked = df_off['inhib_blocked']
x_on = np.hstack((df_off['x'][:39],df_on['x'][38:]))
y_on = np.hstack((df_off['inhib_blocked'][:39],df_on['inhib_on'][38:]))

#plot data
plt.figure()
plt.plot(x_blocked,y_blocked,'k--')
plt.plot(x_on,y_on,'k')

plt.xlabel('Time [ms]')
plt.ylabel('Voltage [mV]')
plt.savefig('plots/lines.png')

pickle.dump([x_on,y_on],open('data/lines_inhibition_on.p','wb'))
pickle.dump([x_blocked,y_blocked],open('data/lines_inhibition_off.p','wb'))
