import domirank as dr
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

N = 2500 #size of network
m = 3 #average number of links per node.
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
A  = np.loadtxt("Crime_Gcc.txt")
G = nx.read_edgelist("Crime_Gcc.txt", )
N = len(G)


#################### insert network hereunder ########################3
GAdj = nx.to_scipy_sparse_array(G)
#GAdj = sp.sparse.random(N,N, density = 0.01, format = 'csr')*1
#GAdj = GAdj @ GAdj.T
#flipping the network direction if it is directed (depends on the interactions of the links...)
if directed:
    GAdj = sp.sparse.csr_array(GAdj.T)
G, node_map = dr.relabel_nodes(G, yield_map = True)
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


fig1 = plt.figure(1)
ourRange = np.linspace(0,1, sigmaArray.shape[0]) 
index = np.where(sigmaArray == sigmaArray.min())[0][-1]

plt.plot(ourRange, sigmaArray)
plt.plot(ourRange[index], sigmaArray[index], 'ro', mfc = 'none', markersize = 10)
plt.xlabel('sigma')
plt.ylabel('loss')


_, ourDomiRankDistribution = dr.domirank(GAdj, analytical = analytical, sigma = sigma) #generate the centrality using the optimal sigma
ourDomiRankAttack = dr.generate_attack(ourDomiRankDistribution) #generate the attack using the centrality (descending)
domiRankRobustness, domiRankLinks = dr.network_attack_sampled(GAdj, ourDomiRankAttack) #attack the network and get the largest connected component evolution

## UNCOMMENT HERE: to compute the analytical solution for the same sigma value (make sure your network is not too big.)
#analyticalDomiRankDistribution = sp.sparse.linalg.spsolve(sigma*GAdj + sp.sparse.identity(GAdj.shape[0]), sigma*GAdj.sum(axis=-1)) #analytical solution to DR
#analyticalDomiRankAttack = dr.generate_attack(analyticalDomiRankDistribution) #generate the attack using the centrality (descending)
#domiRankRobustnessA, domiRankLinksA = dr.network_attack_sampled(GAdj, analyticalDomiRankAttack) #attack the network and get the largest connected component evolution

#generating the plot
fig2 = plt.figure(2)
ourRangeNew = np.linspace(0,1,domiRankRobustness.shape[0])
plt.plot(ourRangeNew, domiRankRobustness, label = 'Recursive DR')
#plt.plot(ourRangeNew, domiRankRobustnessA, label = 'Analytical DR') #UNCOMMENT HERE to plot the analyitcal solution
plt.legend()
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.show()
