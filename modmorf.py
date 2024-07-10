#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
import copy
from scipy import sparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from morphopy.computation import file_manager as fm
from morphopy.neurontree import NeuronTree as nt
import pandas as pd
import networkx as nx


# ## Define morphology functions

# In[ ]:


#function to define number of discrete positions  
def get_number_of_nodes(Adjacency):
    """
    Returns 
        NumberNodes: number of positions/nodes
    args
        Adjacency = adjacecny matrix representing tree (bidirectional)
    """
    NumberNodes = ((Adjacency.get_shape())[1])
    return NumberNodes


# In[ ]:


#function to determine Node connection number

def get_edge_number(Adjacency):
    """
    Returns 
        columnsum: matrix of shape (N,1) with node connection type: a node is a branchpoint(3), endpoint(1) or middle-point(2)
        (Done by summing columns (so only works for unweighted matrices where connected nodes = 1))
    args
        Adjacency: adjacency matrix representing tree (bidirectional)
    """
    columnsum = (np.sum(Adjacency, axis=1)) # columnsum is a numpy.matrix of shape(N,1) # axis=0 is rows/axis =1 is column
    
    return columnsum


# In[ ]:


# define the parent and children nodes adjacency matrices 
#-- need to simplify ChildrenWeighed (copy from children instead of new matrix? and - maybe don't need arg N)

def relations_nodes(Adjacency):
    """
    Returns 
        parent adjacency matrix, children adjacency matrix 
    args
        Adjacency: adjacency matrix representing tree (bidirectional)
    
    NOTE! need to look at the matrix in rows - ie: node 0 is row 0 so node 0 has 0 parent and 1 child 
          if look at it in columns you get the opposite (chilren/parents are switched)
    """
    Parent = sparse.csr_matrix(sparse.tril(Adjacency))        # matrix containing parents of node (=1) (node is row, parents are in column)
    Children = sparse.csr_matrix(sparse.triu(Adjacency))      # matrix containing children of node (=1) (node is row, children are in column)
    
    return Parent, Children



# In[ ]:


def child_matrix_weighed(Children,nodes, branchpoints): 
    """
    Returns 
        ChildrenWeighed: weighed children matrix (weighed means that nodes dostream of branches haea  weight of 0.5 instead of 1)
    args
        Adjacency: adjacency matrix representing tree
        nodes: number of nodes (likely defined as N in simulation scipt)
        branchpoints: number of branchpoints (likely defined as BP in smilaiotn script)
    
    NOTE! need to look at the matrix in rows - ie: node 0 is row 0 so node 0 has 0 parent and 1 child 
    if look at it in columns you get the opposite (chilren/parents are switched)
    """

    ChildrenWeighed = sparse.csr_matrix((nodes, nodes))
    for i in range(nodes):
        if i in branchpoints:
            for j in range(Children[i].nnz):
                ChildrenWeighed[i,(Children[i].indices[j])] = Children[i,Children[i].indices[j]]/Children[i].nnz
        else:
            ChildrenWeighed[i,(Children[i].indices)] = 1
    
    return ChildrenWeighed


# In[ ]:


def child_matrix_weighed2(Children,nodes, branchpoints): 
    """
    Returns 
        ChildrenWeighed: weighed children matrix (weighed means that nodes dostream of branches haea  weight of 0.5 instead of 1)
    args
        Adjacency: adjacency matrix representing tree
        nodes: number of nodes (likely defined as N in simulation scipt)
        branchpoints: number of branchpoints (likely defined as BP in smilaiotn script)
    
    NOTE! need to look at the matrix in rows - ie: node 0 is row 0 so node 0 has 0 parent and 1 child 
    if look at it in columns you get the opposite (chilren/parents are switched)
    """
    ChildrenWeighed = sparse.csr_matrix(Children, dtype=np.float64, copy=True)

    for i in branchpoints:
        for j in range(Children[i].nnz):
            index = i,(Children[i].indices[j])
            ChildrenWeighed[index] =  Children[index]/Children[i].nnz
        
    return ChildrenWeighed


# In[ ]:


def xy_coordinates(NeuronTree):
    """
    returns 
      coordinates: dictionary of xy coordinates for each node - {node0: array([x, y]), ...}
    arg
      NeuronTree: morphopy class - use 'Axon'for simulation
    """
    coordinates = NeuronTree.get_node_attributes('pos')
    
    for i in range(len(coordinates)):
        coordinates[i] = np.delete(coordinates[i],-1) # getting rid of z coordinate
    
    return coordinates


# In[ ]:


def number_branches_per_BOid(ChBrPt, BrOrd, maxBrOrd):
    """
    returns 
        dictionary: dictionary of number of banches for eahc branch order -{BOid: number, ....}
    args
        ChBrPt = use CBP - array containing children of branching points 
        BrOrd = use BO - dict containing branch orde of all nodes
        maxBrOrd = use maxBO - highest branch order 
        
    """
    dictionary = dict.fromkeys(range(0, maxBrOrd), 0)
    dictionary[0] = 1
    for i in ChBrPt.reshape(-1):
        dictionary[BrOrd[i]] += 1
    
    return dictionary


# In[ ]:


def number_endpoints_per_BOid(endpoint, BrOrd, maxBrOrd):
    """ 
    returns 
        dictionary of nuber of endpoints for eahc branch order -{BOid: numeber, ....}
    args
        endpoint = use EP - array containing children of branching points 
        BrOrd = use BO - dict containing branch orde of all nodes
        maxBrOrd = use maxBO - highest branch order 
    """
    dictionary = dict.fromkeys(range(0, maxBrOrd), 0)
    for i in endpoint: 
        dictionary[BrOrd[i]] +=1
    
    return dictionary

