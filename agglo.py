import pandas as pd
from itertools import combinations
import csv
import numpy as  np
import collections
from operator import itemgetter
import re
from scipy.spatial import distance
import scipy
import heapq
import operator


#data input
with open('cho.csv', 'rb') as f:
    reader = csv.reader(f)
    your_database = list(reader)

#data processing
for i in range(0,386): 
    your_database[i] = map(float, your_database[i])

filter_col = lambda lVals, iCol: [[x for i,x in enumerate(row) if i!=iCol] for row in lVals]
preserved_database=your_database
your_database=filter_col(your_database,1)




#initial clusters
clusters_list=np.array(your_database)


#calculate smilarity
def find_pairwise_similarity(a):
    pair_sim = scipy.spatial.distance.pdist(a[:, 1:16], 'euclidean')    
    pair_sim_matrix= scipy.spatial.distance.squareform(pair_sim,'tomatrix') 
    addcol=a[:, :1]
    pair_sim_matrix_revised = np.c_[addcol, pair_sim_matrix]
    return pair_sim_matrix_revised
    
     
pairwise_similarity_result= find_pairwise_similarity(clusters_list)
abcd=pairwise_similarity_result
#for i in range(0, 385):
#    for j in range(0, 386):
#      if abcd[i][j]==0.47936729133306538:
#          print 'i is '+ str(i)
      
i=0
#STEP 1
print 'STEP 1'
print 'pairwise_similarity_result in matrix form' 
#print pairwise_similarity_result[0]


pairwise_similarity_result_list=pairwise_similarity_result.tolist()

most_sim_pair_clusters={}

def find_most_similar_pair(a):
    pair_list=[]
    N=np.shape(a)[1]
    print np.shape(a)
   
    #print a[0] 
    arr=a[:, 1:N]
    min_val=arr[arr>0].min()
    print 'Min value: '+str(min_val)
    pair_set=[(index, row.index(min_val)) for index, row in enumerate(arr.tolist()) if min_val in row]
    
   
    print 'Pair is '+str(pair_set) 
    print a[69][80]
    print a[79][70]
    print a[78][385]
    print a[384][79]
    if pair_set[0][0]<pair_set[0][1]:
       c=pair_set[0][1]+1
       r=pair_set[0][0]
      
      
    else:
        c=pair_set[1][1]+1
        r=pair_set[0][0]
       
       
    #print r
    #print c-1
    actual_cluster1_index=a[r][0]
    actual_cluster2_index=a[c-1][0]
    #actual_cluster2_index=
    pair_list.append(r)
    pair_list.append(c)
    most_sim_pair_clusters[min_val]=str(actual_cluster1_index)+'/'+str(actual_cluster2_index)
    
    return pair_list
    
#STEP2
print 'STEP 2: FIND PAIR'
find_most_similar_pair_result=find_most_similar_pair(pairwise_similarity_result)
xyz=pairwise_similarity_result[69][80]

#STEP 3
print 'STEP 3: DELETE THE PAIR and ADD  a merged cluster'
def remove_clusters(c, d):
    new_list=[]
    N=np.shape(d)[1]
    print 'delete this pair of index '+str(c)
    print N
    for i in range(1,N):
       
        
        #not including the column values that will be deleted later
        if (i!=c[1]+1 and i!=c[0]+1):
           if d[c[0]][i]<d[c[1]][i]:
            #print str(d[c[0]][i])+'<'+str(d[c[1]][i])
            new_list.append(d[c[0]][i])
           if d[c[0]][i]>d[c[1]][i]:
             #print str(d[c[0]][i])+'>'+str(d[c[1]][i])
             new_list.append(d[c[1]][i])
     
        i=i+1 
    print 'no. of iterations skipping 2 columns '+str(i-1)+'='+str(len(new_list))
    last_index=len(new_list)
    #new_list.insert(last_index,0.0)
    #print 'new list after appending 0.0 is '+str(len(new_list))
  
    print 'actual row and columns are: '
    if c[0]<c[1]:
        #remove both rows and columns one by one-- axis=0:row,    axis=1:col
        print 'delete row :'+str(c[0])
        print 'delete column : '+str(c[1])
        print 'before removing '
        print d[69][80]
        
        d=np.delete(d, c[0], 0)
        print d[69][80]
        d=np.delete(d, c[1], 1)
        
        print 'now we are 1 row and column less '
        print 'now delete another row: '+str( c[1]-2)
        print 'now delete another column :'+ str(c[0])
        
        d=np.delete(d, c[1]-2, 0)
        d=np.delete(d, c[0], 1)
        print 'double check '
        print d[69][80]
        print d[78][70]
    else:
        print ' wrong ! check again'
    print 'After removing from pairwise_similarity_result '  
    print  np.shape(d)
    
    #prepare new row
    new_list.insert(0,69*80)
    print 'length after adding sequence '
    print len(new_list)
    
    a= np.array([new_list])
    print 'before transpose'

    print np.shape(a)
    #prepare new column
    a_transpose=np.transpose(a)
    print 'lenght of orig list '+ str(len(a_transpose))
    a_transpose1=np.delete(a_transpose, 0, 0)
    print 'lenght after remove '+ str(len(a_transpose1))
    a_transpose2=np.append(a_transpose1, 0.0)
    print 'lenght of orig list '+ str(len(a_transpose2))
    
    #
    print 'after transpose '
    print np.shape(a_transpose2)
    #print a_transpose2
    a_transpose3 = a_transpose2.reshape(-1,1)
    print np.shape(a_transpose3)
    #add a new row for the merged cluster
    e = np.append(d, a, axis=0)
    print 'after adding 1 row to distance matrix'
    print np.shape(e)
    # 
    print 'after adding 1 column to distance matrix'
    f= np.append(e, a_transpose3, axis=1)
    print np.shape(f)
    #print 'F[0]'
    #print f[384]
    return f
print 'before removing from pairwise_similarity_result '    
print np.shape(pairwise_similarity_result)
remove_clusters_result=remove_clusters(find_most_similar_pair_result, pairwise_similarity_result)





pairwise_similarity_result=remove_clusters_result
print 'second iteration'




find_most_similar_pair_result1=find_most_similar_pair(pairwise_similarity_result)
print pairwise_similarity_result[78][385]
#remove_clusters_result1=remove_clusters(find_most_similar_pair_result1, pairwise_similarity_result)
#addcol=a[:, :1]
#pair_sim_matrix_revised = np.c_[addcol, pair_sim_matrix]
     


