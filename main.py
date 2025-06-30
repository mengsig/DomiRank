import src.DomiRank as dr
from src.utils.NetworkUtils import ( 
    relabel_nodes,
    generate_attack,
    network_attack_sampled,
)
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

########## Figure Parameters ##########
use_latex = False
save_plots = True

if save_plots:
    import os
    os.makedirs("figs", exist_ok = True)

########### FIGURE STUFF ###############
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 35.61 * .5**(.5 * A)])
#Convert to true to use latex
if use_latex:
    plt.rc('text.latex', preamble=r'\usepackage{lmodern}')
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 14})
########################################

N = 10000 #size of network
m = 4 #average number of links per node.
analytical = False #if you want to use the analytical method or the recursive definition
directed = False
seed = 42
np.random.seed(seed)

##### #RANDOMIZATION ######

#for random results
seed = np.random.randint(0, high = 2**32-1)

#for deterministic results
#seed = 42

#setting the random seed
np.random.seed(seed)
##### END OF RANDOMIZATION #####

############## IMPORTANT!!!! Here you can create whatever graph you want and just comment this erdos-renyi network out ############
#G = nx.fast_gnp_random_graph(N, 2*m/N, seed = seed, directed = directed) #####THIS IS THE INPUT, CHANGE THIS TO ANY GRAPH #######

# Using real network "Crime_Gcc"
networkName = "Crime_Gcc.txt"
G = nx.read_edgelist(f"Networks/{networkName}", )
N = len(G)


#################### insert network hereunder ########################3
GAdj = nx.to_scipy_sparse_array(G)
#GAdj = sp.sparse.random(N,N, density = 0.01, format = 'csr')*1
#GAdj = GAdj @ GAdj.T
#flipping the network direction if it is directed (depends on the interactions of the links...)
if directed:
    GAdj = sp.sparse.csr_array(GAdj.T)
G, node_map = relabel_nodes(G, yield_map = True)
#Here we find the maximum eigenvalue using the DomiRank algorithm and searching the space through a golden-ratio/bisection algorithm, taking advantage of the fast divergence when sigma > -1/lambN
t1 = time.time()
lambN = dr.find_eigenvalue(GAdj, maxIter = 500, dt = 0.01, checkStep = 25) #sometimes you will need to change these parameters to get convergence
t2 = time.time()
#IMPORTANT NOTE: for large graphs, comment out the lines below (23-26), along with lines (32-33).
#Please comment the part below (23-26) & (32-33) if you don't want a comparison with how fast the domirank eigenvalue computation is to the numpy computation.

print(f'\nThe found smallest eigenvalue was: lambda_N = {lambN}')
print(f'\nOur single-threaded algorithm took: {t2-t1}s')

#note, if you just perform dr.domirank(GAdj) and dont pass the optimal sigma, it will find it itself.
sigma, sigmaArray = dr.optimal_sigma(GAdj, analytical = analytical, endVal = lambN) #get the optimal sigma using the space (0, -1/lambN) as computed previously
print(f'\n The optimal sigma was found to be: {sigma*-lambN}/-lambda_N')


fig, ax = plt.subplots()
ourRange = np.linspace(0,1, sigmaArray.shape[0]) 
index = np.where(sigmaArray == sigmaArray.min())[0][-1]

ax.plot(ourRange, sigmaArray)
ax.plot(ourRange[index], sigmaArray[index], 'ro', mfc = 'none', markersize = 10)
ax.set_xlabel('sigma')
ax.set_ylabel('area under LCC curve')
fig.set_tight_layout(True)
if save_plots:
    fig.savefig("figs/optimal_sigma.png", dpi = 300)


_, ourDomiRankDistribution = dr.domirank(GAdj, analytical = analytical, sigma = sigma) #generate the centrality using the optimal sigma
ourDomiRankAttack = generate_attack(ourDomiRankDistribution) #generate the attack using the centrality (descending)
domiRankRobustness, domiRankLinks = network_attack_sampled(GAdj, ourDomiRankAttack) #attack the network and get the largest connected component evolution

## UNCOMMENT HERE: to compute the analytical solution for the same sigma value (make sure your network is not too big.)
#analyticalDomiRankDistribution = sp.sparse.linalg.spsolve(sigma*GAdj + sp.sparse.identity(GAdj.shape[0]), sigma*GAdj.sum(axis=-1)) #analytical solution to DR
#analyticalDomiRankAttack = generate_attack(analyticalDomiRankDistribution) #generate the attack using the centrality (descending)
#domiRankRobustnessA, domiRankLinksA = network_attack_sampled(GAdj, analyticalDomiRankAttack) #attack the network and get the largest connected component evolution

#generating the plot
fig2, ax2 = plt.subplots()
ourRangeNew = np.linspace(0,1,domiRankRobustness.shape[0])
ax2.plot(ourRangeNew, domiRankRobustness)#, label = 'Recursive DR')
#ax2.plot(ourRangeNew, domiRankRobustnessA, label = 'Analytical DR') #UNCOMMENT HERE to plot the analyitcal solution
#ax2.legend()
ax2.set_xlabel('fraction of nodes removed')
ax2.set_ylabel('largest connected component')
fig2.set_tight_layout(True)
if save_plots:
    fig2.savefig("figs/llc_curve.png", dpi = 300)
plt.show()
