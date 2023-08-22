import domirank as dr
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time

N = 5000 #size of network
m = 2 #average number of links per node.

############## Here you can create whatever graph you want and just comment this erdos-renyi network out ############
G = nx.fast_gnp_random_graph(N, 2*m/N, directed = True)
#################### insert network hereunder ########################3

GAdj = nx.to_scipy_sparse_array(G)
G, node_map = dr.relabel_nodes(G, yield_map = True)

t1 = time.time()
lambN = dr.find_eigenvalue(GAdj, maxIter = 1000, dt = 0.5, checkStep = 100)
t2 = time.time()
#IMPORTANT NOTE: for large graphs, comment out the lines below (23-26), along with lines (32-33).
#Please comment the part below (23-26) & (32-33) if you don't want a comparison with how fast the domirank eigenvalue computation is to the numpy computation.
GAdjNumpy = nx.to_numpy_array(G)
t3 = time.time()
lambNAnalytical = np.linalg.eigvals(GAdjNumpy).min().real
t4 = time.time()


print(f'\nThe found smallest eigenvalue was: {lambN}')
print(f'The true smallest eigenvalue is: {lambNAnalytical}')
print(f'\nOur single-threaded algorithm took: {t2-t1}s')
print(f'Numpys multi-threaded algorithm took: {t4-t3}s')
print(f'Our Algorithm is {(t4-t3)/(t2-t1)}x faster, even though it is single-threaded')

alpha, alphaArray = dr.optimal_alpha(GAdj, lambN) #get the optimal alpha using the space (0, -1/lambN) as computed previously

fig1 = plt.figure(1)
ourRange = np.linspace(0,1, alphaArray.shape[0]) 
plt.plot(ourRange, alphaArray)
plt.xlabel('sigma')
plt.ylabel('loss')


_, ourDomiRankDistribution = dr.domirank(GAdj, alpha) #generate the centrality using the optimal alpha
ourDomiRankAttack = dr.generate_attack(ourDomiRankDistribution) #generate the attack using the centrality (descending)
domiRankRobustness, domiRankLinks = dr.network_attack_sampled(GAdj, ourDomiRankAttack) #attack the network and 
#get the largest connected component evolution

#generating the plot
fig2 = plt.figure(2)
ourRangeNew = np.linspace(0,1,domiRankRobustness.shape[0])
plt.plot(ourRangeNew, domiRankRobustness)
plt.xlabel('fraction of nodes removed')
plt.ylabel('largest connected component')
plt.show()
