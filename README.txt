This is the code used for the paper:
"DomiRank Centrality: revealing structural fragility of complex networks via node dominance" 
by 
Engsig et. al.

If you use this repository, please cite the following manuscript. --> 
https://arxiv.org/pdf/2305.09589.pdf


In order to run the following code please in your terminal (preferably in a conda environment):

$pip install -r requirements.txt

Thereafter you should be ready to use all the files in the domirank.py module (;

Change G to any network you want (networkx), or import any network and turn it into a scipy.sparse.csr_array() data structure. This will make sure the code runs flawlessly. 

Moreover, in the domirank.domirank() function, if you only pass the adjacency matrix (sparse) as an input, it will automatically compute the optimal sigma. However, you can also pass individual arguments, in order to create domiranks that will damage the network such that it is difficult to recover from, or, to simply, understand dynamics for high sigma (competition).

Finally, the network can be attacked according to any strategy, using the following function. domirank.network_attack_sampled(GAdj, attackStrategy), where GAdj is the adjacency matrix as a scipy.sparse.csr_array(), and the attack strategy is the ordering of the node removals (node-id). The node-id ordering can be generated from the centrality array by using the function domirank.generate_attack(centrality), where, centrality is an array of the centrality-distribution, ordered from (least to greatest in terms of node-id).

To see the latest version, please see the updated version: https://github.com/mengsig/DomiRank

Or, feel free to contact me at: marcus.w.engsig@gmail.com

Enjoy! (:

By: Marcus Engsig
