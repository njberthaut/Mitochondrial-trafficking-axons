#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from __future__ import division
import copy
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# ## Define functions for stats on simulation arrays (for errors in optimisation)

# In[ ]:


def states_to_mobility(states_array):
    """
    converts states array into mobility array
    can take any shape array
    
    returns 
        mobility_array - array of mobile/immobile timepoints (1D or 2D), 0 = immobile / 1 = mobile
    args 
        states_array - array of states timepoints (1D: i.e. svector, 2D: i.e. S)
    """
    
    mobility_array = np.copy(states_array)
    mobility_array[states_array == 1] = 0 # pause -> immobile
    mobility_array[states_array == 2] = 1 # retrograde -> mobile
    mobility_array[states_array == 3] = 1 # anterograde -> mobile
                                             # off track -> immobile (0 -> 0)
    return mobility_array

def mob_immob_transitions_lump(mobility_array): ## old version, calculated in lump, not per track
    """
    returns 
        tr_immob_immob, tr_immob_mob, tr_mob_immob, tr_mob_mob - probabilities of transitionning between immobile and mobile states 
    
    args 
        mobility_array: array of mobile/immobile timepoints (1d or 2d)
    """
    after_immob = []
    after_mob = []
    
    for i in range (len(mobility_array)): #for each run 
        for j in range(len(mobility_array[i])-1): #for each timepoint apart from last timepoint 
            if mobility_array[i,j] == 0:                   # if particle is immobile at timepoint
                after_immob.append(mobility_array[i,j+1])    # add following mobility state to after_immob array 
            if mobility_array[i,j] == 1:                   # if particle is mobile at timepoint
                after_mob.append(mobility_array[i,j+1])     # add following mobility state to after_mob array 
    
    if len(after_immob) == 0: 
        tr_immob_immob = 0
        tr_immob_mob = 0  
    else: 
        tr_immob_immob = after_immob.count(0)/len(after_immob)
        tr_immob_mob = after_immob.count(1)/len(after_immob)
    
    if len(after_mob) == 0: 
        tr_mob_immob = 0
        tr_mob_mob = 0  
    else: 
        tr_mob_immob = after_mob.count(0)/len(after_mob)
        tr_mob_mob = after_mob.count(1)/len(after_mob)
    
    return tr_immob_immob, tr_immob_mob, tr_mob_immob, tr_mob_mob

def mob_immob_transitions(mobility_array): ## new versio per track
    """
    calculate transition probabilities between immobile and mobile states (per particle track)
    
    returns 
        tr_immob_immob, tr_immob_mob, tr_mob_immob, tr_mob_mob - probabilities of transitionning between immobile and mobile states 
    
    args 
        mobility_array: array of mobile/immobile timepoints (1d or 2d)
    """
    tr_from_immob = np.empty((len(mobility_array),2)) #initialise arrays of transition probabilities - each row is a track, each column is a state)
    tr_from_mob = np.empty((len(mobility_array),2))
    
    for i in range (len(mobility_array)): #for each particle run/track
        after_immob_i = []
        after_mob_i = []
        
        for j in range(len(mobility_array[i])-1): #for each timepoint apart from last timepoint 
            if mobility_array[i,j] == 0:                   # if particle is immobile at timepoint
                after_immob_i.append(mobility_array[i,j+1])    # add following mobility state to after_immob array 
            if mobility_array[i,j] == 1:                   # if particle is mobile at timepoint
                after_mob_i.append(mobility_array[i,j+1])     # add following mobility state to after_mob array 
    
    
        if len(after_immob_i) != 0:
            tr_from_immob[i,:] = np.array([after_immob_i.count(0), after_immob_i.count(1)])/len(after_immob_i)
        else: 
            tr_from_immob[i,:] = [np.nan, np.nan]
        if len(after_mob_i) != 0:
            tr_from_mob[i,:] = np.array([after_mob_i.count(0), after_mob_i.count(1)])/len(after_mob_i)
        else: 
            tr_from_mob[i,:] = [np.nan, np.nan]
    
    mean_tr_from_immob = np.nanmean(tr_from_immob, axis=0) 
    mean_tr_from_mob = np.nanmean(tr_from_mob, axis=0) 
    
    tr_immob_immob = mean_tr_from_immob[0] 
    tr_immob_mob = mean_tr_from_immob[1] 
    tr_mob_immob = mean_tr_from_mob[0]  
    tr_mob_mob = mean_tr_from_mob[1]  
    
    return tr_immob_immob, tr_immob_mob, tr_mob_immob, tr_mob_mob


# In[ ]:


def forward_backward_states(states_array):

    """
    defines forward and backward direction for each particle
    forward is defined as the direction the particle takes on its first move and backward the opposite
    
    returns 
        states array with antero/retro states conveted to forward/backward (4 = backward,  5= forward)
    args
        states_array (1D or 2D)
        
    """
    fb_array = np.copy(states_array)                  #4 = backward,  5= forward

    if states_array.ndim == 1: # for 1d array
        
        if 2 not in states_array and 3 not in states_array: 
            fb_array = np.copy(states_array)
        else:

            for i in range(len(states_array)):
                if states_array[i] in [2,3]: #if particle is mobile at timepoint
                    idx = i 
                    fb_array[idx] = 5
                    break

            if states_array[idx] == 2: # if first movement was retrograde
                for i in range(idx+1, len(states_array)):
                    if states_array[i] == 2:    #if movement is retrograde
                        fb_array[i] = 5           #then direction is forward
                    if states_array[i] == 3:              #if moveemnt is anterograde
                        fb_array[i] = 4           #then direction is backward


            if states_array[idx] == 3:            #if first movement was anterograde 
                for i in range(idx+1, len(states_array)):
                    if states_array[i] == 2:    #if movement is retrograde
                        fb_array[i] = 4           #then direction is backward
                    if states_array[i] == 3:              #if moveemnt is anterograde
                        fb_array[i] = 5           #then direction is forward


    if states_array.ndim == 2:  # for a 2D array, performs the count on each row individually 

        for j in range(len(states_array)):         # for each run 
            if 2 not in states_array[j] and 3 not in states_array[j]:
                fb_array[j] = np.copy(states_array[j])
            else:
            
                for i in range(len(states_array[j])):     # for each timepoint in single mito run
                    if states_array[j,i] in [2,3]:
                        idx = i
                        fb_array[j,idx] = 5    
                        break

                if states_array[j,idx] == 2: # if first movement was retrograde
                    for i in range(idx+1, len(states_array[j])):
                        if states_array[j,i] == 2:    #if movement is retrograde
                            fb_array[j,i] = 5           #then direction is forward
                        if states_array[j,i] == 3:              #if moveemnt is anterograde
                            fb_array[j,i] = 4           #then direction is backward


                if states_array[j,idx] == 3:            #if first movement was anterograde 
                    for i in range(idx+1, len(states_array[j])):
                        if states_array[j,i] == 2:    #if movement is retrograde
                            fb_array[j,i] = 4           #then direction is backward
                        if states_array[j,i] == 3:              #if moveemnt is anterograde
                            fb_array[j,i] = 5           #then direction is forward

    return fb_array


def forward_backward_transitions(fb_array): 
    """
    calculates transition probabilies between forward, backward, pause, off-track and immobile (off-track and pause combined) states
    
    returns 
        dict_fb_transitions - diciionary of transition prbabiliities with keys in format e.g. 'tr_f_b' 'tr_f_o' 'tr_f_immo' ...
    args 
        fb_array - array of mobile/immobile timepoints (2D)
    """
    
    tr_from_f = np.empty((len(fb_array),4)) #initialise arrays of transition probabilities - each row is a track, each column is a state)
    tr_from_b = np.empty((len(fb_array),4))
    tr_from_p = np.empty((len(fb_array),4))
    tr_from_o = np.empty((len(fb_array),4))
    tr_from_immostate = np.empty((len(fb_array),2)) 
    
    for i in range (len(fb_array)):  #for each mito track
        after_forward_i = []           #initiliase transition lists
        after_backward_i = []
        after_paused_i = []
        after_off_i = []
        after_immostate_i = []
        
        for j in range(len(fb_array[i])-1):  #fill transition lists 
            if fb_array[i,j] == 0:
                after_off_i.append(fb_array[i,j+1])
            if fb_array[i,j] == 1:
                after_paused_i.append(fb_array[i,j+1])
            if fb_array[i,j] == 4:
                after_backward_i.append(fb_array[i,j+1])
            if fb_array[i,j] == 5:
                after_forward_i.append(fb_array[i,j+1])
            if fb_array[i,j] == 0 or fb_array[i,j] == 1:
                after_immostate_i.append(fb_array[i,j+1])
    
    
        if len(after_forward_i) != 0:
            tr_from_f[i,:] = np.array([after_forward_i.count(0), after_forward_i.count(1), after_forward_i.count(4), after_forward_i.count(5)])/len(after_forward_i)
        else: 
            tr_from_f[i,:] = [np.nan, np.nan, np.nan, np.nan]

        if len(after_backward_i) != 0:
            tr_from_b[i,:] = np.array([after_backward_i.count(0), after_backward_i.count(1), after_backward_i.count(4), after_backward_i.count(5)])/len(after_backward_i)
        else: 
            tr_from_b[i,:] = np.array([np.nan, np.nan, np.nan, np.nan])

        if len(after_off_i) != 0:
            tr_from_o[i,:] = np.array([after_off_i.count(0), after_off_i.count(1), after_off_i.count(4), after_off_i.count(5)])/len(after_off_i)
        else:
            tr_from_o[i,:] = np.array([np.nan, np.nan, np.nan, np.nan])

        if len(after_paused_i) != 0: 
            tr_from_p[i,:] = np.array([after_paused_i.count(0), after_paused_i.count(1), after_paused_i.count(4), after_paused_i.count(5)])/len(after_paused_i)
        else:
            tr_from_p[i,:] = np.array([np.nan, np.nan, np.nan, np.nan])
        
        if len(after_immostate_i) != 0:
            #tr_from_immostate[i,:] = np.array([after_immostate_i.count(4), after_immostate_i.count(5)])/(sum(np.array(after_immostate_i)==4)+sum(np.array(after_immostate_i)==5)) #here we normalised not by all tr fom immo but by transtions from immo->mobile (not in use anymore)
            tr_from_immostate[i,:] = np.array([after_immostate_i.count(4), after_immostate_i.count(5)])/len(after_immostate_i)
        else:
            tr_from_immostate[i,:] = np.array([np.nan, np.nan])
        
        
    tr_from_f_immo = np.sum(tr_from_f[:, [0,1]], axis=1) # 1D array, still includes the nan values (array of transition probabilities)
    tr_from_b_immo = np.sum(tr_from_b[:, [0,1]], axis=1) # 1D array, still includes the nan values
    
    mean_tr_from_f = np.nanmean(tr_from_f, axis=0) 
    mean_tr_from_b = np.nanmean(tr_from_b, axis=0)
    mean_tr_from_p = np.nanmean(tr_from_p, axis=0)
    mean_tr_from_o = np.nanmean(tr_from_o, axis=0)
    mean_tr_from_immostate = np.nanmean(tr_from_immostate, axis=0)
    
    tr_f_immo = np.nanmean(tr_from_f_immo)
    tr_b_immo = np.nanmean(tr_from_b_immo)
    
    tr_f_o = mean_tr_from_f[0]
    tr_f_p = mean_tr_from_f[1]
    tr_f_b = mean_tr_from_f[2]
    tr_f_f = mean_tr_from_f[3]
    
    tr_b_o = mean_tr_from_b[0]
    tr_b_p = mean_tr_from_b[1]
    tr_b_b = mean_tr_from_b[2]
    tr_b_f = mean_tr_from_b[3]
    
    tr_p_o = mean_tr_from_p[0]
    tr_p_p = mean_tr_from_p[1]
    tr_p_b = mean_tr_from_p[2]
    tr_p_f = mean_tr_from_p[3]
    
    tr_o_o = mean_tr_from_o[0]
    tr_o_p = mean_tr_from_o[1]
    tr_o_b = mean_tr_from_o[2]
    tr_o_f = mean_tr_from_o[3]
    
    tr_immo_b = mean_tr_from_immostate[0]
    tr_immo_f = mean_tr_from_immostate[1]

    
    dict_fb_transitions = {'tr_f_o':tr_f_o, 'tr_f_p':tr_f_p, 'tr_f_b':tr_f_b, 'tr_f_f':tr_f_f, 'tr_b_o':tr_b_o, 'tr_b_p':tr_b_p, 'tr_b_b':tr_b_b, 'tr_b_f':tr_b_f, 'tr_o_b':tr_o_b, 'tr_o_f':tr_o_f, 'tr_o_o':tr_o_o, 
                           'tr_o_p':tr_o_p, 'tr_p_b':tr_p_b, 'tr_p_f':tr_p_f, 'tr_p_o':tr_p_o, 'tr_p_p':tr_p_p, 'tr_f_immo':tr_f_immo, 'tr_b_immo':tr_b_immo, 'tr_immo_b':tr_immo_b, 'tr_immo_f':tr_immo_f}
    
    
    return dict_fb_transitions


# In[ ]:


def count_consecutive(arr, n):
    """
    counts consecutive specified value in an array 
    
    returns 
        consecutive_array - counts of consecutive specified value 
    
    args
       arr - array 
       n - number/item to count 
    """
    if arr.ndim == 1: 
        # pad a with False at both sides for edge cases when array starts or ends with n
        d = np.diff(np.concatenate(([False], arr == n, [False])).astype(int)) # difference array
        # subtract indices when value changes from False to True from indices where value changes from True to False
        consecutive_array = np.flatnonzero(d == -1) - np.flatnonzero(d == 1)
    
    if arr.ndim == 2:  # for a 2D array, performs the count on each row individually 
        consecutive_array = np.array([], dtype=int)
        
        for i in range (len(arr)):
            # pad a with False at both sides for edge cases when array starts or ends with n
            d = np.diff(np.concatenate(([False], arr[i] == n, [False])).astype(int))
            # subtract indices when value changes from False to True from indices where value changes from True to False
            consecutive_array = np.append(consecutive_array, (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)))

    return consecutive_array


def immobile_lengths(mobility_array,tstep):
    """
    computes length of time of all immobile bouts (for all particles)
    
    returns 
        immob_lengths_sec - array of immobile bouts lengths (seconds)
    
    args
        mobility_array - array (1D or 2D) of immobile 
        tstep - timestep (float)
    """
    immob_lengths = count_consecutive(mobility_array, 0)
    immob_lengths_sec = (np.copy(immob_lengths))*tstep       #transform values into seconds 
    return immob_lengths_sec


def mobile_lengths(mobility_array,tstep):
    mob_lengths = count_consecutive(mobility_array, 1)
    mob_lengths_sec = (np.copy(mob_lengths))*tstep
    return mob_lengths_sec


def fb_lengths(fb_array,tstep):
    fw_lengths = count_consecutive(fb_array, 5)
    forward_lengths_sec = (np.copy(fw_lengths))*tstep       #transform values into seconds 
    
    bw_lengths = count_consecutive(fb_array, 4)
    backward_lengths_sec = (np.copy(bw_lengths))*tstep       #transform values into seconds 
    
    return forward_lengths_sec, backward_lengths_sec


# ## Define functions for post-simulation outputs

# In[ ]:


# to improve - can make R optional?
def convert_vector (vector, dictionary, t, R):
    """
    convert node ID using other node definition contained in a dictionary 
    
    returns
        vector2 = vector with nodes ID converted to new id i.e. BO id or dist from soma 
                  (1d of length t / or 2d of shape (R,t) according to input)
    
    args 
        vector = vector of position sequence to be converted, can be 1d or 2d  i.e. xvector or X
        dictionary = reference dictionary containing info needed to change vector values (i.e. 'BO' branch order ID of each node)
        t = number of timepoints
        R = number of particles/runs

    """
    if vector.ndim == 1:                        #if input vector is 1d vector
        vector2 = np.empty(t, dtype=float)          #initialise new 1d vector
    
        for i in range(t):                       #convert each node into its new id (ie. its BOid if using BO dictionary)
            if vector[i]>=0:
                vector2[i] = dictionary[vector[i]]
            else:
                vector2[i] = -99999
    
    else:                                      #if input vector is 2d
        vector2 = np.empty((R,t), dtype=float)      #initialise new 2d vector
        
        for i in range(R):
            for j in range(t):
                if vector[i][j]>=0:
                    vector2[i][j] = dictionary[vector[i][j]]  #convert each node into its new id (ie. its BOid if using BO dictionary)
                else:
                    vector2[i][j] = -99999
    return vector2


# In[ ]:


def count_visits(vector, nodes, run):
    """
    to get the number of visits to a node or BOid (per single run or for all runs) 
    i.e. number of timesteps the mito has spent at each node/BOid
    
    returns 
        visits = dictionary of number of visits per node or BOid
    args
        vector = vector of position sequence i.e. X or xvector
        nodes = N if lookig at nodes id, maxBO if looking at BO id
        run = run number (single value using an index), to choose all runs - use None
        
    """
    if vector.ndim == 1:                         # if input vector is 1d (single mito run)
        visits = Counter(vector)
    else:                                        # if input vector is 2d (each row is a different mito run)
        if run == None:                           # if no run number is specified
            visits = Counter(vector.flatten())     # count number of visits for all mito runs  
        else:                                     # if a run number is specified 
            visits = Counter(vector[run])            #slice row of vector and count number of visits for that run
    
    for i in range(nodes):                  # give value of '0' to all the positions that don't have any visits
        if visits.get(i) is None:            
            visits[i] = 0
    
    return visits


# In[ ]:


def count_mitos_spec(t, nodes, vector, pos_list= None, dist_dict=None, interval=None):
    '''
    count mitos at sepcific positions for each timepoint. (several positions and sum results)
    
    returns:
        simcount = array of number of mitos at lumoed positions over time of simulation
    
    argument
        pos_list 
        t = numebr of timepoints
        vector = Xresult

        if using distances rather than node positions
        dist_dict = distfrosoma
        interval = [lower_bound, upper_bound]

        global variable - N (number of positions)

    '''
    if dist_dict == None:
        pos = list(set(pos_list)) # unique elements in list (in case of repeats)
        
        if any(element > nodes for element in pos) == True:
            warnings.warn("No corresponding positions (node out of tree bounds).", UserWarning)        
        
    else: # conevrt distance fomro soma into nodes
        pos_list = []
        
        for key, value in dist_dict.items():
            if interval[0] <= value <= interval[1]:
                pos_list.append(key)
        
        pos = list(set(pos_list))        
        
        if len(pos)==0:
            warnings.warn("No corresponding positions (distance out of tree bounds).", UserWarning)
        
    count = np.empty((len(pos),t))
    for j in range(len(pos)): #loop through relevant positions
        for i in range(t):  #for each timepoint (tp), find number of mitos at position pos
            count[j,i] = np.count_nonzero(vector[:,i] == pos[j])

    sumcount = np.sum(count,axis=0)
    
    return sumcount


# In[ ]:


## at the moment the subsequent normalise function only works for a single timepoint
# tp as list

def count_mitos(vector, nodes, t, tp):
    """ 
    returns
       dictionary of number of mitos at each node or at each BOid at eahc timepoint 
       {timepoint: Counter{BOid/position: number mitos ..., ... }}
    
    args
        vector = position vector/matrix - needs to be 2d (can also be done with BO vector)
        tp is a list of timepoints that want to return  (if tp = None, returns coutnfor all timepoints
        nodes = N (for nodes) or maxBO (for BO id)
        t = umber of timepoints 
    """
    number = {} # initialise an empty dictionary
    
    if tp == None:                             # count number of mitos per node or per BOid 
        number = {}
        for i in range(len(vector[1,:])):
            number[i] = Counter(vector[:,i])
        
        for i in range(t):      
            for j in range(nodes):
                if number[i].get(j) is None: # all the positions that are not occupied  = 0
                    number[i][j] = 0
       
    
    else:
        if len(tp)==0:
            number = Counter(vector[:,tp])    # count number of mitos per node or BOid at specified timetpoint
            
            for i in range(nodes):      
                if number.get(i) is None: # all the positions that are not occupied  = 0
                    number[i] = 0
            
        else:
            number = {} 
            for i in tp:
                number[i] = Counter(vector[:,i])
                
                for j in range(nodes):
                    if number[i].get(j) is None: # all the positions that are not occupied  = 0
                        number[i][j] = 0                
        
    return number


# In[ ]:


# tp as list

def count_mitos_no0(vector, tp):
    """ 
    returns
       dictionary of number of mitos at each node or at each BOid at eahc timepoint 
       {timepoint: Counter{BOid/position: number mitos ..., ... }}
    
    args
        vector = position vector/matrix - needs to be 2d (can also be done with BO vector)
        tp is a list of timepoints that want to return , if tp = None, returns coutnfor all timepoints
        nodes = N (for nodes) or maxBO (for BO id)
    """
    number = {} # initialise an empty dictionary
    
    if tp == None:                             # count number of mitos per node or per BOid 
        number = {}
        for i in range(len(vector[1,:])):
            number[i] = Counter(vector[:,i])
       
    
    else:
        if len(tp)==0:
            number = Counter(vector[:,tp])    # count number of mitos per node or BOid at specified timetpoint

            
        else:
            number = {} 
            for i in tp:
                number[i] = Counter(vector[:,i])
        
    return number


# In[ ]:


##maybe instead of having a standalone normalise function, should normalise inside the previous functions with an argument nomalise = True )

def normalise_count_BOid(countdataBO, BrOrdposN, maxBrOrd):
    """
    retuns 
        array of normalised counted data (i.e. number of mitos per BOid etc) per number of nodes with same BOid
    args 
        countadataBO = counted data from a vector converted to BOid (previous required functions: vector_to_BOvector & count_...)
        BrOrdposN =  number of nodes per BOid /use BOposN
        maxBrOrd = highest branch order / use maxBO 
    """
    normalised = np.empty(maxBrOrd+1)      # initialise array
    
    for i in range(maxBrOrd+1):
        normalised[i] = countdataBO[i]/BrOrdposN[i]        # divide by number of positions that have the same BO id 
    
    normalised = normalised/sum(normalised)*100 # to get percentage
    
    return normalised 


# In[ ]:


def normalise_count_BOid_nested(countdataBO, BrOrdposN, maxBrOrd,t):
    """
    returns 
        nested array of normalised counted data (i.e. number of mitos per BOid etc) per number of nodes with same BOid for nested dict like {tp:{0: x , 1:y ...}
    args 
        countdataBO = counted data from a vector converted to BOid (previous required functions: vector_to_BOvector & count_...)
        BrOrdposN =  number of nodes per BOid /use BOposN
        maxBrOrd = highest branch order / use maxBO 
        t = number of timepoints 
    """
    normalised = np.empty((t, maxBrOrd+1))      # initialise array
    
    for i in range(t):
        for j in range(maxBrOrd+1):
            normalised[i][j] = countdataBO[i][j]/BrOrdposN[j]        # divide by number of positions that have the same BO id 
    
        normalised[i] = normalised[i]/sum(normalised[i])*100 # to get percentage
    
    return normalised 


# In[ ]:


def bin_tree_graph_density(countermitos, arrtps, seg, currentdist, newdist):
    
    resample = int(newdist/currentdist) # number of nodes to be 'lumped' together
    newdict = {}

    for index in arrtps:
        newdict[index] = {}
        #newdict[index][0] = countermitos[index][0]

        for i in range(len(seg)): # for each segment 
            if any(countermitos[index][key] != 0 for key in range(seg[i][0]+1, seg[i][1])): # if there are any mitochondria in that segment 
                num1ind = seg[i][0]    #seg[i][0] + 1
                num2ind = seg[i][0] + resample -1 #seg[i][0] + resample 

                while num2ind <= seg[i][1]:
                    summ = 0

                    for m, value in countermitos[index].items():
                        if m >= num1ind and m <= num2ind:
                            summ += value
                    
                    central_position = (num1ind + num2ind) // 2
                    newdict[index][central_position] = summ
                    
                    num1ind += resample 
                    num2ind += resample
                

            else: # else fill segment's binned count with 0 
                num1ind = seg[i][0] 
                num2ind = seg[i][0] + resample -1

                while num2ind <= seg[i][1]:
                    central_position = (num1ind + num2ind) // 2
                    newdict[index][central_position] = 0  
                    
                    num1ind += resample 
                    num2ind += resample

    return newdict


# In[ ]:


#version 2 - longer
def bin_tree_graph_density_2(countermitos, arrtps, seg, currentdist, newdist):
    '''
    returns
        dictionary of summed number of mitos per segement (for plotting)
    
    args
        countersmito = dictionary of numebr of mitochondria at each X position (numbermitos)
        arrtps = array fo timepoints to convert dct for (tpi)
        index = index of tpi array
        seg = segments_list 
        currentdist = current sampling distqnce 
        new dist = new sampling 'distance'/bin size
    '''
    resample = int(newdist/currentdist) # number fo nodes to be 'lumped'together
    newdict = {}
    for index in arrtps:
        newdict[index] = {}
        newdict[index][0] = countermitos[index][0]
        for i in range(len(seg)): # for each segment 
            num1ind = seg[i][0]+1
            num2ind = seg[i][0]+resample 
            while num2ind <=seg[i][1]:
                rangeind = list(range(num1ind,num2ind+1))
                summ = 0
                for m, value in countermitos[index].items():
                    if m >= num1ind and m <= num2ind:
                        summ += value
                for j in rangeind:
                    newdict[index][j] = summ

                num1ind += resample 
                num2ind += resample

    return(newdict)


# In[ ]:


## not sure what this does 
def number_mitos_endpoints(endpoints, mitocount, t):
    '''
    retuns 
        array ????
    args
        endpoints - list/array fo endpoints (EP)
        mitocount = dict of count of mitos per branchpoints at each timepoint-  numbermitos 
        t = timepoints
    '''
    arrsum = np.zeros(t, dtype=int)
    for x in endpoints:
        arr = []
        for tp in range(t): 
            arr.append(mitocount[tp][x])
        arr = np.array(arr)
        arrsum += arr 

    return(arrsum)


# In[ ]:


def number_mitos_endpoint_um(endpoints, mitocount, sampledist, um, t):
    '''
    returns 
        array ????
    
    args 
        um = number of microns to be considered at endpoint
        sampledist = tree sampling distacne used (ie 0.2)
        endpoints - list/array fo endpoints (EP)
        mitocount = dict of count of mitos per branhpoints at each timepoint-  numerbmitos 
        t =number of timepoints
    '''
    nodes = int(um/sampledist) #number of nodes to make 1um 
    
    arrsum = np.zeros(t, dtype=int)
    for x in endpoints:    # each endpoint 
        xrange = x - np.array(range(nodes)) # select range 
        for i in range(len(xrange)):
            arr = []
            for tp in range(t): 
                arr.append(mitocount[tp][xrange[i]])
            arr = np.array(arr)
            arrsum += arr 

    return(arrsum)


# In[ ]:


def dict_0_to_none(mydict):
    """
    returns 
        dictionary with no 0 values
    args 
        mydict = dict to transform
    """
    newdict = mydict.copy()
    for i, v in newdict.items():
        if v == 0 :
            newdict[i] = None
    return newdict


# In[ ]:


def dict_no0(mydict):
    """
    returns 
        dictionary with no 0 values
    args 
        mydict = dict to transform
    """
    newdict = {x:y for x,y in mydict.items() if y!=0}
    return newdict


# In[ ]:


def nested_dict_no0(mydict):
    """
    returns 
        nested dictionary with no 0 values
    args 
        mydict = nested dict to transform (1 nest only)
    """
    newdict = {}
    for i in  list(mydict.keys()):
        newdict[i] = {x:y for x,y in mydict[i].items() if y!=0}
    return newdict


# In[ ]:


def latency_endpoint(X, endpoints):
    """
    returns 
        array of latencies for eahc mito to reach an endpoint 
    args 
    """
    latencies = np.empty(len(X))
    for i in range(len(X)):
        occ = []
        for o in endpoints: 
            indexes_endpoints = np.where(X[i] == o)
            if len(indexes_endpoints[0]) != 0:
                occ.append(indexes_endpoints[0][0])
        if not occ: 
            latencies[i] = np.NaN
        else: 
            latencies[i] = min(occ)
    return latencies

