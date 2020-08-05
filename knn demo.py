import csv
import numpy as  N
from scipy.spatial import distance
import copy




def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
			
def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax

# read the data
with open('project3_dataset3_train.csv', 'rb') as f:
    reader = csv.reader(f)
    your_database = list(reader)
    
set_database=copy.deepcopy(your_database) 

with open('project3_dataset3_test.csv', 'rb') as f:
    reader = csv.reader(f)
    test_database = list(reader)
    
set_database=copy.deepcopy(your_database) 
set_database1=copy.deepcopy(test_database)

#
#uniqstr=list(zip(*set_database)[4])   
#uniqstr=list(N.unique(uniqstr))
##
#i=0
#stringdict=dict()
#for ustr in uniqstr:
#    stringdict[ustr]=i
#    i=i+1
###ulist=list(set(set_database))
###    
#for item in your_database:
#    un_str=item[4]
#    item[4]=stringdict[un_str]
#    

    
########################another method    
2#for item in your_database:
#    if  item[4]=='Present':
#        item[4]='1.0'
#    if item[4]=='Absent':
#        item[4]='0.0'
#    


r=N.shape(your_database)[0]
r1=N.shape(test_database)[0]

for i in range(0, r):
    your_database[i]=map(float, your_database[i])
    

for i in range(0, r1):
    test_database[i]=map(float, test_database[i])

#your_database_copy=copy.deepcopy(your_database)
your_database_copy=copy.deepcopy(your_database)    
minmax = dataset_minmax(your_database_copy)
normalize_dataset(your_database_copy, minmax)
train_database=copy.deepcopy(your_database_copy)


#your_database_copy=copy.deepcopy(your_database)
test_database_copy=copy.deepcopy(test_database)    
minmax = dataset_minmax(test_database_copy)
normalize_dataset(test_database_copy, minmax)
test1_database=copy.deepcopy(test_database_copy)

#xx=[]
#xx.extend(your_database_copy[4:10])
#print xx[1][0]
#xx=[]
#xx.extend(your_database_copy[11:20])
#print xx[1][0]
#xx=[]
#xx.extend(your_database_copy[24:300])
#print xx[1][0]
#xx=[]
#xx.extend(your_database_copy[34:40])
#print xx[1][0]


def partition1(lst,n):
 division = len(lst) / float(n)
 return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def partition2(database, split_percent):
 x=list(partition1(database, split_percent))

 for i in xrange(0,len(x)):
  test = x[i]
  train =[]

 for j in xrange(0,i):
  train += x[j]
 for p in xrange(i+1,len(x)):
   train +=x[p]
 return train, test
 
#k=15
k = input('Enter the value of k nearest neighbors to be considered: ')

    

def findknearest_neighbors(test_tuple,training_set):
    flist=list()
    neighborlist=list()   
    for item in training_set:
        lst=list()
        dst=distance.euclidean(test_tuple[0:len(test_tuple)-1],item[0:len(item)-1])
        #print dst
        label=item[len(item)-1]
        lst.append(dst)
        lst.append(label)
        flist.append(lst)
    flist.sort()
    for i in range(0,k):
        neighborlist.append(flist[i])
    return neighborlist

def compute_kNN(testingset,trainpartition ):
 resultlabelset=list()  
 for testtuple in testingset:
    label0=0
    label1=0
    k_nearest_neighbors=findknearest_neighbors(testtuple, trainpartition)
    #finding the majority for labels
    for neighbor in k_nearest_neighbors:
        if(neighbor[1]==0):
            label0=label0+1
        if(neighbor[1]==1):
            label1=label1+1
            
        if label1>label0:
            resultlabel=1.0
        else:
            resultlabel=0.0
    resultlabelset.append(resultlabel)
 return resultlabelset
         
        
#predictresult=compute_kNN(test_partition, train_partition)


def accuracy(predicted, actual):
    a=0
    b=0
    c=0
    d=0
    cl=int(N.shape(your_database_copy)[1])-1
    last=list(zip(*actual)[cl])
    #print last
    acc=0
    for each in range(0, len(predicted)):
        if last[each]==1 and predicted[each]==1:
            a=a+1
        if last[each]==1 and predicted[each]==0:
            b=b+1
            
        if last[each]==0 and predicted[each]==1:
            c=c+1
            
        if last[each]==0 and predicted[each]==0:
            d=d+1
    accuracy=(a+d)/float(a+b+c+d)
    precision=a/float(a+c)
    recall=a/float(a+b)
    fmeasure=2*a/float((2*a)+b+c)
    
    return accuracy,precision,recall,fmeasure
    
    


#print accuracy(predictresult, test_partition)

predictresult=compute_kNN(test1_database, train_database)
accuracy1,precision_total1,recall_total1,f1_1 =accuracy(predictresult, test1_database)

accuracy1=accuracy1*100

print 'The accuracy of the required Data set is: '+str(accuracy1)
print 'The precision of the required Data set is: '+str(precision_total1)
print 'The recall of the required Data set is: '+str(recall_total1)
print 'The f1 Measure of the required Data set is: '+str(f1_1)
  
    
#def kfold():
#    acc=list()
#    precision=list()
#    recall=list()
#    f1=list()
#    split_percent=10
#    i=0
#    each_split=int(round(len(your_database_copy)*(0.1)))
#    j=i+each_split
#    test_partition= your_database_copy[i:j]
#    train_partition=your_database_copy[0: i]
#    train_partition.extend(your_database_copy[j: len(your_database_copy)])
#    for i in range(0, split_percent):
#        j=i+each_split
#        test_partition= your_database_copy[i:j]
#        train_partition=[]
#      
#        if i==0:
#            train_partition=your_database_copy[57: len(your_database_copy)]
#
#        if i >0:
#            train_partition.extend(your_database_copy[0: i])
#            x=your_database_copy[j: len(your_database_copy)]
#            #print '---------part 1 -------------'
#            #print x[20][0]
#            #print x[30][0]
#            train_partition.extend(x)
#            #print '---------part 2 -------------'
#            #print train_partition[i+20][0]
#            #print train_partition[i+30][0]
#            #x=[]
#        #print N.shape(train_partition)
#        train_rows=N.shape(train_partition)[0]
#        train_col=N.shape(train_partition)[1]
#        predictresult=compute_kNN(test_partition, train_partition)
#        accuracy1,precision_total1,recall_total1,f1_1 =accuracy(predictresult, test_partition)
#        print 'Accuracy: ' + str(accuracy1)
#        acc.append(accuracy1)
#        precision.append(precision_total1)
#        recall.append(recall_total1)
#        f1.append(f1_1)
#    return acc,precision,recall,f1,predictresult
#        
#acclist=list()        
#acc_lst,precision_lst,recall_lst,f1_lst,pred=kfold()    

#def k_fold_cross_validation(X, K, randomise = False):
#	"""
#	Generates K (training, validation) pairs from the items in X.
#
#	Each pair is a partition of X, where validation is an iterable
#	of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
#
#	If randomise is true, a copy of X is shuffled before partitioning,
#	otherwise its order is preserved in training and validation.
#	"""
#	if randomise: from random import shuffle; X=list(X); shuffle(X)
#	for k in xrange(K):
#		training = [x for i, x in enumerate(X) if i % K != k]
#		validation = [x for i, x in enumerate(X) if i % K == k]
#		print training[30][0]
#		yield training, validation
#
#X = your_database
#for training, validation in k_fold_cross_validation(X, K=7):
#	for x in X: assert (x in training) ^ (x in validation), x
#	



