########## Here are the associated DomiRank functions #############
import numpy as np
import scipy as sp
import networkx as nx
from utils.NetworkUtils import (
    generate_attack,
    network_attack_sampled,
)

######## Beginning of domirank stuff! ####################

def domirank(G, analytical = True, sigma = -1, dt = 0.1, epsilon = 1e-5, maxIter = 1000, checkStep = 10):
    '''
    G is the input graph as a (preferably) sparse array.
    This solves the dynamical equation presented in the Paper: "DomiRank Centrality: revealing structural fragility of
complex networks via node dominance" and yields the following output: bool, DomiRankCentrality
    Here, sigma needs to be chosen a priori.
    dt determines the step size, usually, 0.1 is sufficiently fine for most networks (could cause issues for networks
    with an extremely high degree, but has never failed me!)
    maxIter is the depth that you are searching with in case you don't converge or diverge before that.
    Checkstep is the amount of steps that you go before checking if you have converged or diverged.
    
    
    This algorithm scales with O(m) where m is the links in your sparse array.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        G = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        G = G.copy()
    if analytical == False:
        if sigma == -1:
            sigma, _ = optimal_sigma(G, analytical = False, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        pGAdj = sigma*G.astype(np.float64)
        Psi = np.ones(pGAdj.shape[0]).astype(np.float64)/pGAdj.shape[0]
        maxVals = np.zeros(int(maxIter/checkStep)).astype(np.float64)
        dt = np.float64(dt)
        j = 0
        boundary = epsilon*pGAdj.shape[0]*dt
        for i in range(maxIter):
            tempVal = ((pGAdj @ (1-Psi)) - Psi)*dt
            Psi += tempVal.real
            if i% checkStep == 0:
                if np.abs(tempVal).sum() < boundary:
                    break
                maxVals[j] = tempVal.max()
                if i == 0:
                    initialChange = maxVals[j]
                if j > 0:
                    if maxVals[j] > maxVals[j-1] and maxVals[j-1] > maxVals[j-2]:
                        return False, Psi
                j+=1

        return True, Psi
    else:
        if sigma == -1:
            sigma = optimal_sigma(G, analytical = True, dt=dt, epsilon=epsilon, maxIter = maxIter, checkStep = checkStep) 
        Psi = sp.sparse.linalg.spsolve(sigma*G + sp.sparse.identity(G.shape[0]), sigma*G.sum(axis=-1))
        return True, Psi
    
def find_eigenvalue(G, minVal = 0, maxVal = 1, maxDepth = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10):
    '''
    G: is the input graph as a sparse array.
    Finds the largest negative eigenvalue of an adjacency matrix using the DomiRank algorithm.
    Currently this function is only single-threaded, as the bisection algorithm only allows for single-threaded
    exection. Note, that this algorithm is slightly different, as it uses the fact that DomiRank diverges
    at values larger than -1/lambN to its benefit, and thus, it is not exactly bisection theorem. I haven't
    tested in order to see which exact value is the fastest for execution, but that will be done soon!
    Some notes:
    Increase maxDepth for increased accuracy.
    Increase maxIter if DomiRank doesn't start diverging within 100 iterations -- i.e. increase at the expense of 
    increased computational cost if you want potential increased accuracy.
    Decrease checkstep for increased error-finding for the values of sigma that are too large, but higher compcost
    if you are frequently less than the value (but negligible compcost).
    '''
    x = (minVal + maxVal)/G.sum(axis=-1).max()
    minValStored = 0
    for i in range(maxDepth):
        if maxVal - minVal < epsilon:
            break
        if domirank(G, False, x, dt, epsilon, maxIter, checkStep)[0]:
            minVal = x
            x = (minVal + maxVal)/2
            minValStored = minVal
        else:
            maxVal = (x + maxVal)/2
            x = (minVal + maxVal)/2
        if minVal == 0:
            print(f'Current Interval : [-inf, -{1/maxVal}]')
        else:
            print(f'Current Interval : [-{1/minVal}, -{1/maxVal}]')
    finalVal = (maxVal + minVal)/2
    return -1/finalVal



############## This section is for finding the optimal sigma #######################

def process_iteration(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling):
    tf, domiDist = domirank(spArray, analytical = analytical, sigma = sigma, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    domiAttack = generate_attack(domiDist)
    ourTempAttack, __ = network_attack_sampled(spArray, domiAttack, sampling = sampling)
    finalErrors = ourTempAttack.sum()
    q.put((i, finalErrors))

def optimal_sigma(spArray, analytical = True, endVal = 0, startval = 0.000001, iterationNo = 100, dt = 0.1, epsilon = 1e-5, maxIter = 100, checkStep = 10, maxDepth = 100, sampling = 0):
    ''' This part finds the optimal sigma by searching the space, here are the novel parameters:
    spArray: is the input sparse array/matrix for the network.
    startVal: is the starting value of the space that you want to search.
    endVal: is the ending value of the space that you want to search (normally it should be the eigenvalue)
    iterationNo: the number of partitions of the space between lambN that you set
    
    return : the function returns the value of sigma - the numerator of the fraction of (\\sigma)/(-1*lambN)
    '''
    if endVal == 0:
        endVal = find_eigenvalue(spArray, maxDepth = maxDepth, dt = dt, epsilon = epsilon, maxIter = maxIter, checkStep = checkStep)
    import multiprocessing as mp
    endval = -0.9999/endVal
    tempRange = np.arange(startval, endval + (endval-startval)/iterationNo, (endval-startval)/iterationNo)
    processes = []
    q = mp.Queue()
    for i, sigma in enumerate(tempRange):
        p = mp.Process(target=process_iteration, args=(q, i, analytical, sigma, spArray, maxIter, checkStep, dt, epsilon, sampling))
        p.start()
        processes.append(p)

    results = [None] * len(tempRange)  # Initialize a results list

    #Join the processes and gather results from the queue
    for p in processes:
        p.join()

    #Ensure that results are fetched from the queue after all processes are done
    while not q.empty():
        idx, result = q.get()
        results[idx] = result  # Store result in the correct order

    finalErrors = np.array(results)
    minEig = np.where(finalErrors == finalErrors.min())[0][-1]
    minEig = tempRange[minEig]
    return minEig, finalErrors
