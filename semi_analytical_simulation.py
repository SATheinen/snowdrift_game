import numpy as np
import matplotlib.pyplot as plt
import os

grid_size = 101
lower_bound = -4
upper_bound = 4
time = 10000
delta_t = -0.001

x_0_space = np.array([-0.3]) #np.linspace(-0.3, 0.3, 11) 
b1_s_space = np.linspace(lower_bound, upper_bound, grid_size)
b2_s_space = np.linspace(lower_bound, upper_bound, grid_size)

b1_s_space = np.sign(b1_s_space) * np.power(10, np.sign(b1_s_space) * b1_s_space)
b2_s_space = np.sign(b2_s_space) * np.power(10, np.sign(b2_s_space) * b2_s_space)

b1_s_space[0] = -np.inf
b1_s_space[-1] = np.inf

b2_s_space[0] = -np.inf
b2_s_space[-1] = np.inf

result = np.empty((grid_size, grid_size))
#ana_result = np.empty((grid_size, grid_size))
EUS_array = np.empty((grid_size, grid_size))
nan_replacer = np.empty((grid_size, grid_size))

def x_dot(x_s, b1_s, b2_s):
    return (b1_s - 1 + (4*b2_s - 2) * x_s)

def x_dotC(x_s, c1_s, c2_s):
    return (-c1_s + 1 + (4 - 2*c2_s) * x_s)

def x_dotdot(b2_s):
    return 2*b2_s

def x(b1_s, b2_s):
    return (1 - b1_s) / (4*b2_s - 2)

x_s = x_0_space[0]
for t in range(time):
    x_s += x_dot(x_s, 1.316, 0.875) * delta_t
    print(x_dot(x_s, 1.316, 0.875) * delta_t) 

EUS = x_dotdot(b2_s_space)

for i in range(len(b1_s_space)):
    EUS_array[i,:] = EUS

for x_0 in x_0_space:

    for i, b1_s in enumerate(b1_s_space):
        for j, b2_s in enumerate(b2_s_space):
                
            x_s = x_0
            for t in range(time):
                x_s += x_dotC(x_s, b1_s, b2_s) * delta_t
            result[j, i] = x_s
            
            # calculate in which direction the gradient runs
            nan_replacer[j, i] = x_dotC(x_0, b1_s, b2_s)
            #ana_result[j, i] = x(b1_s, b2_s)

    result = np.where(result < -100., -100., result)
    result = np.where(result > 100., 100., result)
    nan_replacer = np.where(nan_replacer < 0., -100., nan_replacer)
    nan_replacer = np.where(nan_replacer > 0., 100., nan_replacer)
    result[np.isnan(result)] = nan_replacer[np.isnan(result)]
    
    EUS_array = np.where(EUS_array > 0, 1, 0)
    #result = np.where((EUS_array == 1) & (np.abs(result) < 10), 0, result)
        
    plt.imshow(result,origin="lower",aspect='auto',cmap='viridis',extent=[0,grid_size,0,grid_size],vmin=-1.0,vmax=1.0)
    plt.colorbar(label='Final x')
    plt.xlabel("bs_1")
    plt.ylabel("bs_2")

    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()

    x_ticks = np.linspace(0, grid_size-1, 11)
    x_tick_labels = np.around(np.linspace(lower_bound, upper_bound, 11), 1)
    plt.xticks(ticks=x_ticks, labels=x_tick_labels)

    y_ticks = np.linspace(0, grid_size-1, 11)
    y_tick_labels = np.around(np.linspace(lower_bound, upper_bound, 11), 1)
    plt.yticks(ticks=y_ticks, labels=y_tick_labels)
     
    tr = grid_size / (upper_bound - lower_bound)
    x = int(0.85*tr + grid_size/2)
    y = int(0.33*tr + grid_size/2)
    print(x,y)

    EUS_x = np.empty(grid_size)
    EUS_x[:] = np.linspace(0, grid_size, grid_size)
    EUS_y = np.empty(grid_size)
    EUS_y[:] = int((np.log10(1.0/(upper_bound - lower_bound) * grid_size) + (grid_size/2)))

    #os.system("rm -rf semi_analytical_plots")
    #os.system("mkdir semi_analytical_plots")

    plt.plot(EUS_x, EUS_y, 'ro', markersize=2)
    #plt.plot(x, y, 'ro', markersize=5)
    #plt.savefig("semi_analytical_plots/x_0=" + str(x_0) + ".png")
    plt.show()
    plt.close("all")
""" 
    ##########################
    # analytical result
    ##########################
    plt.imshow(ana_result,origin="lower",aspect='auto',cmap='viridis',extent=[0,grid_size,0,grid_size],vmin=-2.0,vmax=2.0)
    plt.colorbar(label='Final x')
    plt.xlabel("bs_1")
    plt.ylabel("bs_2")

    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()

    x_ticks = np.linspace(0, grid_size-1, 11)
    x_tick_labels = np.around(np.linspace(lower_bound, upper_bound, 11), 1)
    plt.xticks(ticks=x_ticks, labels=x_tick_labels)

    y_ticks = np.linspace(0, grid_size-1, 11)
    y_tick_labels = np.around(np.linspace(lower_bound, upper_bound, 11), 1)
    plt.yticks(ticks=y_ticks, labels=y_tick_labels)
     
    #plt.plot(x,y,'ro',markersize=5)
    #plt.show()

    plt.close("all")
"""
