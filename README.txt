This is the code used for the paper:
"DomiRank Centrality: revealing structural fragility of complex networks via node dominance" 
by 
Engsig et. al.

If you use this repository, please cite the following manuscript. --> 
https://arxiv.org/pdf/2305.09589.pdf


In order to run the following code please in your terminal:

$pip install -r requirements.txt

Thereafter you should be ready to use all the files in the domirank.py module (;

Change G to any network you want (networkx). 

IMPORTANT NOTE: In the main.py you can change to any network you want, and if you don't want to compare the speed of the new most negative eigenvalue search, then please just comment out the numpy part (lines (23-26) & (32-33)), as it will take around 100 seconds for a matrix of 10000x10000 and it scales with the cube. However, the domirank algorithm scales linearly with the number of links in the network, so it is applicable up to matrices of 24,000,000 x 24,000,000 and it takes a couple hundred seconds depending on the parameters.


Enjoy! (:

By: Marcus Engsig
