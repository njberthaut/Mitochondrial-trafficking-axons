#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
from numpy import random,argwhere,argmax,linspace,logspace,var,mean,tile,dot
import copy
from numpy.random import randint
from scipy.stats import norm
from scipy.stats import nbinom
from scipy import sparse
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import scipy.io as sio


# ## Define simulation functions

# In[ ]:


def update(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity):
    """
    Updates the particle's position and state according to a set of probabilities
    
    returns
        xnext = updated position (int)
        snext = updated state (int) 
    
    args 
        x - current position
        s - current state 
        states - array of possible states (off = 0, pause = 1, retrograde = 2, anterograde = 3)
        Tr - states probability transition matrix for continous and branching-points
        TrEP - states probability transition matrix for endpoints (no possible retrograde mvmt)
        TrSP - states probability transition matrix for startpoint (no possible anterograde mvmt)
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points 
        velocity - (1 - average speed) (in nodes per timestep)  
   
    """
    #STEP 1: update next state (according to current state)
    if x == 0:                                             # if partcicle is at start point use TrSP to update next state
        snext = random.choice(states, p = (TrSP[s,:]))         
    elif x in endpoints:                                   # if partice is at an endpoint use TrEP to update next state
        snext = random.choice(states, p = (TrEP[s,:]))         
    else:                                                  #else use Tr to update next state
        snext = random.choice(states, p = (Tr[s,:]))         


    #STEP 2: update next position (according to current position and next state)
    if snext == 0 or snext == 1:                                                              # if particle is in off-track or paused state
        xnext = x                                                                                 # update next position same as current position (stationary)
    
    elif snext == 2:                                                                          # if particle is in retrograde state 
        numbernodes = 1 + random.poisson(velocity) #choose how many nodes to move from poisson distribution, with lambda = (1 - average speed) 
        for i in range(numbernodes):
            if x == 0:
                tempx = 0
            else:
                if x in ChBrPt:
                    tempx = (parent[x,:]).indices
                else: 
                    tempx = x-1 
                x = tempx
        xnext = tempx    # xnext gets value of last value of tempx              

    elif snext == 3:                                                                          # if particle is in anterograde state     
        numbernodes = 1 + random.poisson(velocity) #choose how many nodes to move from poisson distribution, with lambda = (1 - average speed)
        for i in range(numbernodes):
            if x in endpoints:
                tempx = x
            else:
                if x in BrPt: 
                    tempx = random.choice(((children[x,:]).indices), p = (children[x,:].data))
                else:
                    tempx = x+1
                x = tempx
        xnext = tempx    # xnext gets value of last value of tempx 
          
    return xnext, snext


# In[ ]:


def update_after_rejoin(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity):
    """
    Updates the particle's position and track state according to a set of probabilities
    only allows anterograde or retrograde movement (no stationary)
    
    returns
        xnext - updated position (int)
        snext - updated state (int) 
    
    args 
        x - current position
        s - current state 
        states - array of possible states (indexes: off = 0, pause = 1, retrograde = 2, anterograde = 3)
        Tr - states probability transition matrix for continous and branching-points
        TrEP - states probability transition matrix for endpoints (no possible retrograde mvmt)
        TrSP - states probability transition matrix for startpoint (no possible anterograde mvmt)
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points 
        velocity - (1 - average speed) (in nodes per timestep) 
   
    """
    #STEP 1: update next state (according to current state)
    if x == 0:                    # if partcicle is at start point use TrSP to update next state (only antero and reto states, normalised)
        snext = random.choice([2,3] , p = [(TrSP[0,2]/(TrSP[0,2]+TrSP[0,3])) , (TrSP[0,3]/(TrSP[0,2]+TrSP[0,3]))])          
    elif x in endpoints:          # if partice is at an endpoint use TrEP to update next state (only antero and reto states, normalised)                              
        snext = random.choice([2,3] , p = [(TrEP[0,2]/(TrEP[0,2]+TrEP[0,3])) , (TrEP[0,3]/(TrEP[0,2]+TrEP[0,3]))])         
    else:                         # else use Tr to update next state (only antero and reto states, normalised)                     
        snext = random.choice([2,3] , p = [(Tr[0,2]/(Tr[0,2]+Tr[0,3])) , (Tr[0,3]/(Tr[0,2]+Tr[0,3]))])         


    #STEP 2: update next position (according to current position and next state)
    if snext == 2:                                                                          # if particle is in retrograde state 
        numbernodes = 1 + random.poisson(velocity) #choose how many nodes to move from poisson distribution, with lambda = (1 - average speed)
        for i in range(numbernodes):
            if x == 0:
                tempx = 0
            else:
                if x in ChBrPt:
                    tempx = (parent[x,:]).indices
                else: 
                    tempx = x-1 
                x = tempx
        xnext = tempx      # xnext gets value of last value of tempx
                         

    elif snext == 3:                                                                          # if particle is in anterograde state     
        numbernodes = 1 + random.poisson(velocity) #choose how many nodes to move from poisson distribution, with lambda = (1 - average speed)
        for i in range(numbernodes):
            if x in endpoints:
                tempx = x
            else:
                if x in BrPt: 
                    tempx = random.choice(((children[x,:]).indices), p = (children[x,:].data))
                else:
                    tempx = x+1
                x = tempx
        xnext = tempx    # xnext gets value of last value of tempx
          
    return xnext, snext


# In[ ]:


def run(t, Tr, TrEP, TrSP, states, parent, children, endpoints, BrPt, ChBrPt, nodes, velocity, xstart, sstart, sinitprob, randx, rands):
    """
    Run simulation for a single particle and save vectors recording position and state along the particle's run.
    nested 'update' and 'update_after_rejoin' functions
    speeding up by sampling duration of off-track states from negative binoamial distribution
    
    returns
        xvector - vector of position sequence for a single run (1D array of length t, dtype=int)
        svector - vector of state sequence for a single run (1D array of length t, dtype=int)
        
    args
        t - number of timesteps
        Tr - states probability transition matrix for continous and branching-points
        TrEP - states probability transition matrix for endpoints (no possible retrograde mvmt)
        TrSP - states probability transition matrix for startpoint (no possible anterograde mvmt)
        states - array of possible states (indexes: off = 0, pause = 1, retrograde = 2, anterograde = 3)
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points split
        nodes - number of nodes in tree (N)
        velocity - (1 - average speed) (in nodes per timestep) 
        
        endpoints,BrPt,ChBrPt - arrays/dictionaries?
        
        randx=False - select True if want random position start for each particle run 
        rands=False - select true if want random state start for each particle run
        xstart - inital position (if not random)
        sstart - inital state (if not random)
        sinitprob - inital states probabilities (ie steady states - likelihood to be in any state at a given time)
    """
    
    #initialise empty vector
    xvector = np.empty(t, dtype=int)
    svector = np.empty(t, dtype=int)
    
    #determine inital state and position
    if randx == True:
        xinit = random.randint(0,nodes) 
    else: 
        xinit = xstart 
    
    if rands == True:
        sinit = random.choice(states, p=sinitprob)
    else: 
        sinit = sstart
    
    # record inital state and position in vectors
    xvector[0] = xinit 
    svector[0] = sinit 
    
    #variables x and s take current (inital) position and state
    x = xinit
    s = sinit
    
    i = 1 # manually increment i by 1 or jump ahead if off track
    
    if sinit == 0: # if started off-track 
        
        rejoin = abs(random.negative_binomial(1, (1-Tr[0,0]))) # sample how long mito stays in off-track state using negative biomial distribution 
                                                                #(had to add abs for extremely high numbers sampled during optimisation - above int32)
        
        if rejoin > (t-1)-i: # if number of timsteps to rejoin is longer than run time, whole sim has same position and off-track state  
            xvector[i:t] = x 
            svector[i:t] = s # (s=0)
            i+=rejoin # update variable i to timepoint of rejoin
        
        else:                # esle fill timepoints before rejoin with same position and off-track state
            xvector[i:i+rejoin-1] = x
            svector[i:i+rejoin-1] = s # (s=0)
            i+=rejoin  # update variable i to timepoint of rejoin

            #update timepoint after rejoin
            xnext, snext = update_after_rejoin(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity)
            xvector[i] = xnext
            svector[i] = snext
            
            #variables x and s take current (inital) position and state
            x = xnext
            s = snext

            i+=1
    
    while i < t:
        
        xnext, snext = update(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity) # update position and state
        
        xvector[i] = xnext
        svector[i] = snext
      
        #variables x and s take current position and state
        x = xnext
        s = snext
        
        i+=1
    
        if s == 0:
            rejoin = abs(random.negative_binomial(1, (1-Tr[0,0]))) # sample how long mito stays in off-track state using negative biomial distribution 
                                                                     #(had to add abs for extremely high numbers sampled during optimisation - above int32)
            
            if rejoin > (t-1)-i: 
                xvector[i:t] = x 
                svector[i:t] = s 
                i+=rejoin
            
            else:
                xvector[i:i+rejoin-1] = x
                svector[i:i+rejoin-1] = s 
                i+=rejoin
        
                xnext, snext = update_after_rejoin(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity)
                xvector[i] = xnext
                svector[i] = snext

                x = xnext
                s = snext
            
                i+=1
        
    return xvector, svector


# In[ ]:


def sim_ensemble(parent, children, simparams, transitionparams, scale=False):
    """
    Run simulation for an ensemble of particles (R) and save vectors recording position and along the particles' runs.
    nested run function
    
    returns
        X = array of shape (R,t) recording positions 
        S = array of shape (R,t) recording states 
        (arrays are 2D where row = particle run, and column = timepoint)
    
    args
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points split
        simparams - dictionary of simulation parameters
        transitionparams - DTYPE??? states transitions probabilities
        scale = True or False - scale transition proabbilites to sum to 1 (needed for optimisiation simualtions)

    """
    #unpack simulation parameters from dictionary: fixed & intiation paramters
    
    t = simparams['numtimesteps'] # number of timesteps
    R = simparams['runs'] # number of particles
    states = simparams['statesarray'] #array of possible states (off = 0, pause = 1, retrograde = 2, anterograde = 3)
    velocity = simparams['adjusted_velocity'] # (1 - average speed) (in nodes per timestep) 
    nodes = simparams['N'] # number of nodes in tree
    endpoints = simparams['EP'] # endpoints #####type?
    BrPt = simparams['BP'] # branching points ####type?
    ChBrPt = simparams['CBP'] # children of branching points ###type?

    xstart = simparams['x0'] # inital position (if not random)
    sstart = simparams['s0'] #  inital state (if not random)
    randx = simparams['randompos'] # random initial position - True or False
    rands = simparams['randomstate'] # random initial state - True or False
    
    #unpack parameters: states transition probabilities parameters (s0 = off-track, s1=pause, s2=retrograde, s3=anterograde)
    s0s0, s0s2, s0s3, s1s1, s1s2, s1s3, s2s0, s2s1, s2s2, s2s3, s3s0, s3s1, s3s2, s3s3 = transitionparams
    
    s0s1 = 0
    s1s0 = 0
    
    #build transition probabilities 2D arrays
    Tr = np.array([[s0s0, s0s1, s0s2, s0s3], [s1s0, s1s1, s1s2, s1s3], [s2s0, s2s1, s2s2, s2s3], [s3s0, s3s1, s3s2, s3s3]], dtype='float')
    TrEP = np.array([[s0s0, s0s1, s0s2+s0s3, 0], [s1s0, s1s1, s1s2+s1s3, 0], [s2s0, s2s1, s2s2+s2s3, 0], [s3s0, s3s1, s3s2+s3s3, 0]], dtype='float')
    TrSP = np.array([[s0s0, s0s1, 0, s0s2+s0s3], [s1s0, s1s1, 0, s1s2+s1s3], [s2s0, s2s1, 0, s2s2+s2s3], [s3s0, s3s1, 0, s3s2+s3s3]], dtype='float')
    
    
    ######## ONLY NECESSARY FOR NON-CONSTRAINED optimiser #######
    #scale transitions probabilities to have probabilites of transitioning from each state sum to 1 
    if scale == True:
        for i in range(len(Tr)):
            Tr[i] = Tr[i]/sum(Tr[i])
            TrEP[i] = TrEP[i]/sum(TrEP[i])
            TrSP[i] = TrSP[i]/sum(TrSP[i])
    ##############################################################
    
    #Inital state 
    if simparams['states_startprob'] == None: # if no given initial states probabilities, obtain steady states of states trnasiiton proababilities using eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(np.asmatrix(Tr).T)
        steady_state_index = np.where(np.isclose(eigenvalues, 1))[0][0]
        steady_state_vector = abs(np.real(eigenvectors[:, steady_state_index]))
        steady_state_vector = np.asarray(steady_state_vector.ravel())[0]
        sinitprob = steady_state_vector/np.sum(steady_state_vector)

    else:                                         # else use given inital states probabilities 
        sinitprob = simparams['states_startprob']
    
    #Initialise 2D vectors
    X = np.empty([R, t], dtype=int)
    S = np.empty([R, t], dtype=int)

    #Run simulation
    for i in range(R):
        xvector, svector = np.array(run(t, Tr, TrEP, TrSP, states, parent, children, endpoints, BrPt, ChBrPt, nodes, velocity, xstart, sstart, sinitprob, randx, rands))
        X[i] = xvector
        S[i] = svector
    
    return X , S


# In[ ]:


def run_snapshot(t, Tr, TrEP, TrSP, states, parent, children, endpoints, BrPt, ChBrPt, nodes, velocity, xstart, sstart, sinitprob, randx, rands, snapshot):
    """
    Run simulation for a single particle and save vectors recording position and state at snapshots (specific timepoints) along the particle's run.
    nested 'update' and 'update_after_rejoin' functions
    speeding up by sampling duration of off track states from 
    
    returns
        xvector - vector of position sequence for a single run (1d array of length t, dtype=int)
        svector - vector of state sequence for a single run (1d array of length t, dtype=int)
        
    args
        t - number of timesteps
        Tr - states probability transition matrix for continous and branching-points
        TrEP - states probability transition matrix for endpoints (no possible retrograde mvmt)
        TrSP - states probability transition matrix for startpoint (no possible anterograde mvmt)
        states - array of possible states (indexes: off = 0, pause = 1, retrograde = 2, anterograde = 3)
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points split
        nodes - number of nodes in tree (N)
        velocity - (1 - average speed) (in nodes per timestep) 
        
        endpoints,BrPt,ChBrPt - arrays/dictionaries?
        
        randx=False - select True if want random position start for each particle run 
        rands=False - select true if want random state start for each particle run
        xstart - inital position (if not random)
        sstart - inital state (if not random)
        sinitprob - inital states probabilities (ie steady states - likelihood to be in any state at a given time)
        
        snapshot - time interval (in timepoints) to record 
    """
    
    #initialise empty vector
    xvector = np.empty(t, dtype=int)
    svector = np.empty(t, dtype=int)
    
    #determine inital state and position
    if randx == True:
        xinit = random.randint(0,nodes) 
    else: 
        xinit = xstart 
    
    if rands == True:
        sinit = random.choice(states, p=sinitprob)
    else: 
        sinit = sstart
    
    # record inital state and position in vectors
    xvector[0] = xinit 
    svector[0] = sinit 
    
    #variables x and s take current (inital) position and state
    x = xinit
    s = sinit
    
    i = 1 # manually increment i by 1 or jump ahead if off track
    
    if sinit == 0: # if started off-track 
        
        rejoin = abs(random.negative_binomial(1, (1-Tr[0,0]))) # sample how long mito stays in off-track state using negative biomial distribution 
                                                                #(had to add abs for extremely high numbers sampled during optimisation - above int32)
        
        if rejoin > (t-1)-i: # if number of timsteps to rejoin is longer than run time, whole sim has same position and off-track state  
            xvector[i:t] = x 
            svector[i:t] = s # (s=0)
            i+=rejoin # update variable i to timepoint of rejoin
        
        else:                # esle fill timepoints before rejoin with same position and off-track state
            xvector[i:i+rejoin-1] = x
            svector[i:i+rejoin-1] = s # (s=0)
            i+=rejoin  # update variable i to timepoint of rejoin

            #update timepoint after rejoin
            xnext, snext = update_after_rejoin(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity)
            xvector[i] = xnext
            svector[i] = snext
            
            #variables x and s take current (inital) position and state
            x = xnext
            s = snext

            i+=1
    
    while i < t:
        
        xnext, snext = update(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity) # update position and state
        
        xvector[i] = xnext
        svector[i] = snext
      
        #variables x and s take current position and state
        x = xnext
        s = snext
        
        i+=1
    
        if s == 0:
            rejoin = abs(random.negative_binomial(1, (1-Tr[0,0]))) # sample how long mito stays in off-track state using negative biomial distribution 
                                                                     #(had to add abs for extremely high numbers sampled during optimisation - above int32)
            
            if rejoin > (t-1)-i: 
                xvector[i:t] = x 
                svector[i:t] = s 
                i+=rejoin
            
            else:
                xvector[i:i+rejoin-1] = x
                svector[i:i+rejoin-1] = s 
                i+=rejoin
        
                xnext, snext = update_after_rejoin(x, s, states, Tr, TrEP, TrSP, parent, children, endpoints, BrPt, ChBrPt, velocity)
                xvector[i] = xnext
                svector[i] = snext

                x = xnext
                s = snext
            
                i+=1
    
    
    xvector = xvector[::snapshot]  
    svector = svector[::snapshot]   
        
    return xvector, svector


# In[ ]:


def sim_ensemble_snapshot(parent, children, simparams, transitionparams, snapshot=1, scale=False): # add parameters argument?
    """
    Run simulation for an ensemble of particles (R) and save vectors recording position and along the particles' runs.
    nested run function
    
    returns
        X = array of shape (R,t) recording positions 
        S = array of shape (R,t) recording states 
        (arrays are 2D where row = 1 particle run, and column = timepoint)
    
    args
        parent - parent matrix (sparse matrix)
        children - children matrix (sparse matrix) - use weighed matrix to acccount for branching-points split
        simparams - dictionary of simulation parameters
        transitionparams - DTYPE??? states transitions probabilities
        scale - True or False - scale transition proabbilites to sum to 1 (needed for optimisiation simualtions)
        
        snapshot - time interval (in timepoints) to record
    """
    #unpack simulation parameters from dictionary: fixed & intiation paramters
    
    t = simparams['numtimesteps'] # number of timesteps
    R = simparams['runs'] # number of particles
    states = simparams['statesarray'] #array of possible states (off = 0, pause = 1, retrograde = 2, anterograde = 3)
    velocity = simparams['adjusted_velocity'] # (1 - average speed) (in nodes per timestep) 
    nodes = simparams['N'] # number of nodes in tree
    endpoints = simparams['EP'] # endpoints #####type?
    BrPt = simparams['BP'] # branching points ####type?
    ChBrPt = simparams['CBP'] # children of branching points ###type?

    xstart = simparams['x0'] # inital position (if not random)
    sstart = simparams['s0'] #  inital state (if not random)
    randx = simparams['randompos'] # random initial position - True or False
    rands = simparams['randomstate'] # random initial state - True or False
    
    #unpack parameters: states transition probabilities parameters (s0 = off-track, s1=pause, s2=retrograde, s3=anterograde)
    s0s0, s0s2, s0s3, s1s1, s1s2, s1s3, s2s0, s2s1, s2s2, s2s3, s3s0, s3s1, s3s2, s3s3 = transitionparams
    
    s0s1 = 0
    s1s0 = 0
    
    #build transition probabilities 2D arrays
    Tr = np.array([[s0s0, s0s1, s0s2, s0s3], [s1s0, s1s1, s1s2, s1s3], [s2s0, s2s1, s2s2, s2s3], [s3s0, s3s1, s3s2, s3s3]], dtype='float')
    TrEP = np.array([[s0s0, s0s1, s0s2+s0s3, 0], [s1s0, s1s1, s1s2+s1s3, 0], [s2s0, s2s1, s2s2+s2s3, 0], [s3s0, s3s1, s3s2+s3s3, 0]], dtype='float')
    TrSP = np.array([[s0s0, s0s1, 0, s0s2+s0s3], [s1s0, s1s1, 0, s1s2+s1s3], [s2s0, s2s1, 0, s2s2+s2s3], [s3s0, s3s1, 0, s3s2+s3s3]], dtype='float')
    
    
    ######## ONLY NECESSARY FOR NON-CONSTRAINED optimiser #######
    #scale transitions probabilities to have probabilites of transitioning from each state sum to 1 
    if scale == True:
        for i in range(len(Tr)):
            Tr[i] = Tr[i]/sum(Tr[i])
            TrEP[i] = TrEP[i]/sum(TrEP[i])
            TrSP[i] = TrSP[i]/sum(TrSP[i])
    ##############################################################
    
    #Inital state 
    if simparams['states_startprob'] == None: # if no given initial states probabilities, otain steady states of states trnasiiton proababilities using eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(np.asmatrix(Tr).T)
        steady_state_index = np.where(np.isclose(eigenvalues, 1))[0][0]
        steady_state_vector = abs(np.real(eigenvectors[:, steady_state_index]))
        steady_state_vector = np.asarray(steady_state_vector.ravel())[0]
        sinitprob = steady_state_vector/np.sum(steady_state_vector)

    else:                                         # else use given inital states probabilities 
        sinitprob = simparams['states_startprob']
    
    #Initialise 2D vectors
    X = np.empty([R, int(t/snapshot)], dtype=int)
    S = np.empty([R, int(t/snapshot)], dtype=int)

    #Run simulation
    for i in range(R):
        xvector, svector = np.array(run_snapshot(t, Tr, TrEP, TrSP, states, parent, children, endpoints, BrPt, ChBrPt, nodes, velocity, xstart, sstart, sinitprob, randx, rands, snapshot))
        X[i] = xvector
        S[i] = svector
    
    return X , S


# In[ ]:


def degrade(decay_mean, posarray, statearray, R, N, tstep):
    '''
    returns position and states array with degraded particles represented by int. -99999
    
    args:
        decay_mean - in seconds
        posarray - 2D array of positions over time (Xresult)
        statearray- 2D array of states over time (Sresult)
        R - number of particle runs
        N - number of timepoints
        tstep - timestep (in seconds)
    '''

    posarray_Deg = np.copy(posarray)

    for i in range((R)):
        degr = int((np.random.exponential(scale=decay_mean))/tstep)
        posarray_Deg[i][degr:N] = -99999
    
    statearray_Deg = np.copy(statearray)
    statearray_Deg[np.where(posarray_Deg == -99999)] = -99999
    
    return posarray_Deg, statearray_Deg

