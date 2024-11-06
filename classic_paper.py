import numpy as np
import matplotlib.pyplot as plt

b1 = 6.
b2 = -0.9
c1 = 4.56
c2 = -1.0
N = 2000
mu = 0.02
sigma = 0.01
time = 75000

# initial population parameters
mean = 0.0
std = 0.0005

# plotting parameters
num_bins = 100

parallel_games = 200

pop = np.random.normal(mean, std, N)

# prevent values to break out of the boundaries
np.where(pop > 1, 1, pop)
np.where(pop < 0, 0, pop)

def B(x, y, b1=b1, b2=b2):
    z = x + y
    return b1 * z + b2 * z**2

def C(x, c1=c1, c2=c2):
    return c1 * x + c2 * x**2

def P(x, y):
    return B(x, y) - C(x)

# calculate alpha
steps = 100
x = np.linspace(0, 1, steps)
y = np.linspace(0, 1, steps)
u = np.linspace(0, 1, steps)

X, Y, U = np.meshgrid(x, y, u)
P_tensor = P(X, U) - P(Y, U)
alpha = np.max(P_tensor)
alpha = np.abs(alpha)

def w(x, y, z, alpha=alpha):
    return (P(y,z) - P(x,z)) / alpha

result = np.empty((time, N))

for t in range(time):
    
    x_index = np.random.randint(0, N, parallel_games)
    y_index = np.random.randint(0, N, parallel_games)
    z_index = np.random.randint(0, N, parallel_games)

    x_trait = pop[x_index]
    y_trait = pop[y_index]
    z_trait = pop[z_index]

    # games
    prob = w(x_trait, y_trait, z_trait)
    test_prob = np.random.rand(parallel_games)
    pos_prob = prob > 0
    neg_prob = prob <= 0
    pos_prob = pos_prob & (test_prob < np.abs(prob))
    neg_prob = neg_prob & (test_prob < np.abs(prob))
    pop[x_index[pos_prob]] = y_trait[pos_prob]
    pop[x_index[neg_prob]] = x_trait[neg_prob]

    # mutations
    num_mutations = int(mu * parallel_games)
    x_mutation = np.random.normal(x_trait[0:num_mutations], sigma, num_mutations)
    x_mutation = np.where(x_mutation > 1, 1, x_mutation)
    x_mutation = np.where(x_mutation < 0, 0, x_mutation)
    pop[x_index[0:num_mutations]] = x_mutation

    result[t, :] = pop

bin_edges = np.histogram_bin_edges(result, num_bins)
heatmap = np.array([np.histogram(row, bins=bin_edges, density=True)[0] for row in result])
heatmap = heatmap/np.max(heatmap)

plt.imshow(heatmap, aspect='auto', cmap='viridis', vmin=0, vmax=0.1)
plt.colorbar(label='Density')
plt.xlabel("Trait value")
plt.ylabel("Time")
plt.gca().invert_yaxis()

x_ticks = np.linspace(0, num_bins-1, 6)
x_tick_labels = [0.0,0.2,0.4,0.6,0.8,1.0]
plt.xticks(ticks=x_ticks, labels=x_tick_labels)

mu_x = np.mean(pop)
print(mu_x)
print(mu_x * (c2/c1))

plt.show()
