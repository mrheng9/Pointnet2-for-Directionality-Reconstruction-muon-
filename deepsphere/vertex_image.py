import numpy as np 
import matplotlib.pyplot as plt
import matplotlib

predict_xyz = np.load('pred_xyz.npy')
true_xyz = np.load('true_xyz.npy')

from scipy.stats import norm

def one_hist2d(pred,true,coord_name):
    plt.figure(figsize=(8, 6))
    plt.hist2d(true,pred, bins=(200,200), norm=matplotlib.colors.LogNorm())
    plt.plot([-17000,17000], [-17000, 17000], label='y=x', color='red', linewidth=0.8)
    plt.axis('equal') 
    plt.xlabel(f'True {coord_name} (mm)', fontweight='bold')
    plt.ylabel(f'Pred {coord_name} (mm)', fontweight='bold')
    plt.title(f'Pred {coord_name} vs True {coord_name}', fontweight='bold')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.colorbar()
    plt.savefig(f'{coord_name}_pred_vs_true.png')
    
def xyz_pred_vs_ture(pred_xyz,xyz):
    one_hist2d(pred_xyz[:,0],xyz[:,0],'x')
    one_hist2d(pred_xyz[:,1],xyz[:,1],'y')
    one_hist2d(pred_xyz[:,2],xyz[:,2],'z')
    
def distance_pdf(distance):
    (mu, sigma)=norm.fit(distance)
    distance_sort=sorted(distance)
    size=len(distance_sort)
    for i in range(size):
        if (i + 1) / size > 0.68:
            quantile68=distance_sort[i - 1]
            break
        elif (i + 1) / size == 0.68:
            quantile68=distance_sort[i]
            break

    plt.figure()
    plt.hist(distance, bins=100, range=(0,1000), color='green', density=True)
    plt.axvline(x=quantile68, color='black', linewidth=2, linestyle='--', label='68%% quantile: %.2f' % quantile68)
    plt.xlim(0,1000)
    plt.legend(frameon=False)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xlabel('Distance mm', fontweight='bold')
    plt.ylabel('P.D.F', fontweight='bold')
    plt.title('Distance P.D.F', fontweight='bold')
    plt.savefig('2.1#distance_pdf.png', dpi=500)
    plt.close()

distance_list=[]
for i in range(true_xyz.shape[0]):
    distance=np.sqrt((true_xyz[i,0]-predict_xyz[i,0])**2
                     +(true_xyz[i,1]-predict_xyz[i,1])**2
                     +(true_xyz[i,2]-predict_xyz[i,2])**2
                     )

    distance_list.append(distance)

distance_pdf(distance_list)
xyz_pred_vs_ture(predict_xyz,true_xyz)
