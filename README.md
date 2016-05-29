# LLC_coding
OpenCV implementation of LLC algorithm (Locality-constrained Linear Coding)
Reference paper: Locality-constrained Linear Coding for Image Classification
https://www.robots.ox.ac.uk/~vgg/rg/papers/wang_etal_CVPR10.pdf

Idea: For a vector V, after K-Nearest Neighbour, M closest codeword are picked to be the sparse coding
representation of V. The definition of distance is defined in equation 4 in the paper
