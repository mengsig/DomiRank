########## Here are the associated DomiRank functions #############
import numpy as np
import scipy as sp
import scipy.sparse
import networkx as nx

########## Here are the general functions needed for efficient dismantling and testing of networks #############

def get_largest_component(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph'''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return G

def relabel_nodes(G, yield_map = False):
    '''relabels the nodes to be from 0, ... len(G).
    1. Yield_map returns an extra output as a dict. in case you want to save the hash-map to retrieve node-id'''
    if yield_map == True:
        nodes = dict(zip(range(len(G)), G.nodes()))
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G, nodes
    else:
        G = nx.relabel_nodes(G, dict(zip(G.nodes(), range(len(G)))))
        return G
    
def get_component_size(G, strong = False):
    '''
    here we get the largest component of a graph, either from scipy.sparse or from networkX.Graph datatype.
    1. The argument changes whether or not you want to find the strong or weak - connected components of the graph'''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if nx.is_directed(G) and strong == False:
            GMask = max(nx.weakly_connected_components(G), key = len)
        if nx.is_directed(G) and strong == True:
            GMask = max(nx.strongly_connected_components(G), key = len)
        else:
            GMask = max(nx.connected_components(G), key = len)
        G = G.subgraph(GMask)
        return len(GMask)        
    elif type(G) == scipy.sparse.csr_array:
        if strong == False:
            connection_type = 'weak'
        else:
            connection_type = 'strong'
        noComponent, lenComponent = sp.sparse.csgraph.connected_components(G, directed = True, connection = connection_type, return_labels = True)
        return np.bincount(lenComponent).max()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type or scipy.sparse.csr array')
        
def get_link_size(G):
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        links = len(G.edges()) #convert to scipy sparse if it is a graph 
    elif type(G) == scipy.sparse.csr_array:
        links = G.sum()
    else:
        raise TypeError('You must input a networkx.Graph Data-Type')
    return links

def remove_node(G, removedNode):
    '''
    removes the node from the graph by removing it from a networkx.Graph type, or zeroing the edges in array form.
    '''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        if type(removedNode) == int:
            G.remove_node(removedNode)
        else:
            for node in removedNode:
                G.remove_node(node) #remove node in graph form
        return G
    elif type(G) == scipy.sparse.csr_array:
        diag = sp.sparse.csr_array(sp.sparse.eye(G.shape[0])) 
        diag[removedNode, removedNode] = 0 #set the rows and columns that are equal to zero in the sparse array
        G = diag @ G 
        return G @ diag
    
def generate_attack(centrality, node_map = False):
    '''we generate an attack based on a centrality measure - 
    you can possibly input the node_map to convert the attack to have the correct nodeID'''
    if node_map == False:
        node_map = range(len(centrality))
    else:
        node_map = list(node_map.values())
    zipped = dict(zip(node_map, centrality))
    attackStrategy = sorted(zipped, reverse = True, key = zipped.get)
    return attackStrategy

def network_attack_sampled(G, attackStrategy, sampling = 0):
    '''Attack a network in a sampled manner... recompute links and largest component after every xth node removal, according to some - 
    G: is the input graph, preferably as a sparse array.
    inputed attack strategy
    Note: if sampling is not set, it defaults to sampling every 1%, otherwise, sampling is an integer
    that is equal to the number of nodes you want to skip every time you sample. 
    So for example sampling = int(len(G)/100) would sample every 1% of the nodes removed'''
    if type(G) == nx.classes.graph.Graph: #check if it is a networkx Graph
        GAdj = nx.to_scipy_sparse_array(G) #convert to scipy sparse if it is a graph 
    else:
        GAdj = G.copy()
    
    if sampling == 0:
        if GAdj.shape[0] < 100:
            sampling = 1
        else:
            sampling = int(GAdj.shape[0]/100)
    N = GAdj.shape[0]
    initialComponent = get_component_size(GAdj)
    initialLinks = get_link_size(GAdj)
    m = GAdj.sum()/N
    componentEvolution = np.zeros((N//sampling + 1))
    linksEvolution = np.zeros((N//sampling) + 1)
    j = 0 
    for i in range(N-1):
        if i % sampling == 0:
            if i == 0:
                componentEvolution[j] = get_component_size(GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j+=1 
            else:
                GAdj = remove_node(GAdj, attackStrategy[i-sampling:i])
                componentEvolution[j] = get_component_size(GAdj)/initialComponent
                linksEvolution[j] = get_link_size(GAdj)/initialLinks
                j+=1
    return componentEvolution, linksEvolution
