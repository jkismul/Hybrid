import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import savgol_filter


channels = 16 #number of electrodes

height_step = 100
heights = np.arange(100,1601,height_step)

images=['data/LFP_inhib_on.jpg','data/LFP_inhib_off.jpg',
        'data/iCSD_inhib_on.jpg','data/iCSD_inhib_on.jpg',]

for i in images:
    img = cv2.imread(i)
# img = cv2.imread('data/inhib_on.jpg')
# img = cv2.imread('data/LFP_inhib_off.jpg')

    img_height = img.shape[0]
    img_step = int(np.floor(img_height/channels))

    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    color_1=np.asarray([0,0,22])
    color_2=np.asarray([255,255,255])

    mask = cv2.inRange(img_rgb,color_1,color_2)
    img_rgb[mask!=0]=(0,0,0)
    img_red = []

    for i in range(channels):
        img_red.append(np.sum(img_rgb[i*img_step,:,:],axis=1).astype(int))
    # img_red.append(np.zeros(img_rgb.shape[1]))#skipping blue, doesnt do much
    max_red = np.max(img_red)
    # red_factor=5./max_red #iCSD
    red_factor=1./max_red #LFP

    # fig = plt.figure(figsize=[9, 4])
    # fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
    # ax_o = fig.add_subplot(121, title="original")
    # ax_m = fig.add_subplot(122, title="filter")
    # ax_o.imshow(img_rgb)
    # ax_m.imshow(mask)
    # plt.show()

    # plt.figure()
    # plt.plot(img_rgb[0,:,0])
    # plt.plot(img_rgb[0,:,1])
    # plt.plot(img_rgb[0,:,2])
    # plt.plot(np.sum(img_rgb[0,:,:],axis=1),'k')
    # plt.show()

    #####above is red part

# img = cv2.imread('data/inhib_on.jpg')
img = cv2.imread('data/LFP_inhib_off.jpg')

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

color_1=np.asarray([12,0,0])
color_2=np.asarray([255,255,255])

mask = cv2.inRange(img_rgb,color_1,color_2)
img_rgb[mask!=0]=(0,0,0)
img_blue=[]
# img_blue.append(np.zeros(img_rgb.shape[1])) #skipping red, doesnt do much

for i in range(channels):
    img_blue.append(np.sum(img_rgb[i*img_step,:,:],axis=1).astype(int))
max_blue = np.max(img_blue)
# blue_factor=5./max_blue #iCSD
blue_factor=4./max_blue
# img_blue = np.sum(img_rgb[0,:,:],axis=1).astype(int)
# fig = plt.figure(figsize=[9, 4])
# fig.subplots_adjust(hspace=0.5, top=0.75, bottom=0.2)
# ax_o = fig.add_subplot(121, title="original")
# ax_m = fig.add_subplot(122, title="filter")
# ax_o.imshow(img_rgb)
# ax_m.imshow(mask)
# plt.show()
#
#
# plt.figure()
# plt.plot(img_rgb[0,:,0])
# plt.plot(img_rgb[0,:,1])
# plt.plot(img_rgb[0,:,2])
# plt.plot(np.sum(img_rgb[0,:,:],axis=1),'k')
# plt.show()

##above is blue part

sumd = np.subtract(np.asarray(img_red),np.asarray(img_blue))
sum_scaled=np.subtract(red_factor*np.asarray(img_red),blue_factor*np.asarray(img_blue))
timeticks = 5
plt_heights = []

# sum_smooth = savgol_filter(sumd,len(sumd[0])-1,7,mode='constant')
sum_smooth = savgol_filter(sumd,15,3,mode='constant')
sum_scaled_smooth = savgol_filter(sum_scaled,15,3,mode='constant')
# img = cv2.imread('data/inhib_on.jpg')
img = cv2.imread('data/LFP_inhib_off.jpg')

img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
x=np.linspace(0,3200,len(sumd[0]))


fig,ax = plt.subplots()
ax.imshow(img_rgb,extent=[0,3200,-3200,0],aspect='equal')
for i in range(channels):
    # plt.plot(x,(-i*200)+sumd[i],color='green',linestyle='dashed',linewidth=0.5)
    # plt.plot(x,(-i*200)+100*sum_scaled[i],color='cyan',linestyle='dashed',linewidth=0.5)

    plt.plot(x,(-i*200)+100*sum_scaled_smooth[i],color='cyan',linestyle='dashed',linewidth=0.5)

    plt_heights.append(-i*200)
plt.yticks(plt_heights,heights)
plt.xticks(np.linspace(0,3200,timeticks),np.linspace(0,20,timeticks))

plt.xlabel('time [ms]')
plt.ylabel('depth [um]')
plt.savefig('plots/laminar_LFP_inhib_off.jpg')

# plt.show()
print('SKALERING FORTSATT TATT PÅ ØYEMÅL! FIX!')
print('SMOOTHING ER IKKE HELT TIPP TOPP')
twentyfive = 40/25
half = int(len(sum_scaled_smooth[0])/twentyfive)
plt.figure()
plt.plot(sum_scaled_smooth[1][:half],'r',label='Supragranular')
plt.plot(sum_scaled_smooth[7][:half],'b',label='Granular')
plt.xticks(np.linspace(0,len(sum_scaled_smooth[0][:half]),6),np.linspace(0,25,6))
plt.xlabel('time [ms]')
plt.ylabel('Voltage [mV]')
plt.legend()
plt.savefig('plots/LFP_25ms_supra_and_granular_blocked.jpg')
