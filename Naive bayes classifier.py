import csv
import numpy as  N
from scipy.spatial import distance
import copy
import math
from random import shuffle



def seperateinto2classes(train_database):
    classA=list()
    classB=list()
    train_database_f=copy.deepcopy(train_database)
    for item in train_database_f:
        if item[-1]==0.0:
            classA.append(item)
        else:
            classB.append(item)
    
 
    return classA,classB
    
    
 #read data 
fname = input('Enter the file name Project3_Dataset1 (1) OR Project3_Dataset2(2): ')
if fname==1:
    with open('project3_dataset1.csv', 'rb') as f:
     reader = csv.reader(f)
     your_database = list(reader)
if fname==2:
    with open('project3_dataset2.csv', 'rb') as f:
     reader = csv.reader(f)
     your_database = list(reader)



set_database=copy.deepcopy(your_database) 

if fname==2:
    uniqstr=list(zip(*set_database)[4])   
    uniqstr=list(N.unique(uniqstr))
#
    i=0
    stringdict=dict()
    for ustr in uniqstr:
     stringdict[ustr]=i
     i=i+1
##ulist=list(set(set_database))
##    
    for item in your_database:
      un_str=item[4]
      item[4]=stringdict[un_str]
    


    
    
####3#3    
#for item in your_database:
#    if  item[4]=='Present':
#        item[4]='1.0'
#    if item[4]=='Absent':
#        item[4]='0.0'
##acgg=list()    



r=N.shape(your_database)[0]

for i in range(0, r):
    your_database[i]=map(float, your_database[i])

your_database_copy=copy.deepcopy(your_database)
#minmax = dataset_minmax(your_database_copy)
#normalize_dataset(your_database_copy, minmax)


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
 

 
#train_partition, test_partition=partition2(your_database, 10)
def compute_NaiveBayes(test_partition,train_partition,test_partition_actual):
     train_rows=N.shape(train_partition)[0]
     train_col=N.shape(train_partition)[1]

     class0,class1=seperateinto2classes(train_partition)


     class0_copy=copy.deepcopy(class0)
     class1_copy=copy.deepcopy(class1)

     class0np=N.asarray(class0_copy)
     class1np=N.asarray(class1_copy)


 
     ###TRainign the model 
     summary_dicty=dict()
     keys=[0,1]
     summary_dicty=dict([(key, []) for key in keys])
#summarydicty[0]

     for i in range(0,int(N.shape(class0np)[1])-1):
         lstt_0=list()
         lstt1_0=list()
    
         lstt_1=list()
         lstt1_1=list()
    
         lstt_0=class0np[:,i]
         lstt_1=class1np[:,i]
    #means=N.mean(lstt0)
    #standard_deviation=N.std(lstt0)
         lstt1_0.append(N.mean(lstt_0))
         lstt1_0.append(N.std(lstt_0))
         summary_dicty[0].append(lstt1_0)
    
         lstt1_1.append(N.mean(lstt_1))
         lstt1_1.append(N.std(lstt_1))
         summary_dicty[1].append(lstt1_1)
    
#testing the model
     strr=list()
     st=0
     for item in test_partition:
       test_tuple=item[0:len(item)-1]
       probability0=1
       probability1=1
       for i in range(0,int(N.shape(class0np)[1])-1):
         mean_0=summary_dicty[0][i][0]
         standard_deviation_0=summary_dicty[0][i][1]
         
         if((2*math.pow(standard_deviation_0,2)))==0.0:
             exponent=1
         else: 
          #print standard_deviation_0        
          exponent = math.exp(-(math.pow(test_tuple[i]-mean_0,2)/(2*math.pow(standard_deviation_0,2))))
	 prb_0=(1 / (math.sqrt(2*math.pi) * standard_deviation_0)) * exponent
         probability0*=prb_0
         #print probability0
        
         mean_1=summary_dicty[1][i][0]
         standard_deviation_1=summary_dicty[1][i][1]
         
         if((2*math.pow(standard_deviation_1,2)))==0.0:
             exponent=1
         else:
          exponent = math.exp(-(math.pow(test_tuple[i]-mean_1,2)/(2*math.pow(standard_deviation_1,2))))
	 prb_1=(1 / (math.sqrt(2*math.pi) * standard_deviation_1)) * exponent
         probability1*=prb_1
         
       #probability of the Data tuple belonging to class0 and 1 respectively
       p0=probability0/float(probability1+probability0)
       p1=probability1/float(probability1+probability0)
       #print  test_partition_actual[st][0:len(test_partition_actual[st])-1]
       st=st+1
       #print item
       #print 'probability of Data tuple belonging to class0: ' +str(p0)
       #print 'probability of Data tuple belonging to class1: ' +str(p1)
       if probability1>probability0:
        label=1.0
        #print label
        
       else:
        label=0.0
        #print label
       strr.append(label)
       #if item[-1]==label:
       # strr.append('yes')
       #else:
       # strr.append('No')
     return strr
     
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
           


def kfold(n_folds):
    #n_folds=5
    precision=list()
    recall=list()
    f1=list()
    scores = list()
    split_percent=n_folds
    deno=100/n_folds
    acc=list()
    #start=len(your_database_copy)/float(split_percent)
    pro=(split_percent*len(your_database_copy))/100
    if pro<1:
        pro=1
    #print pro
    kkk=0
    each_split=pro
    #print each_split
    j=kkk+each_split
    #print j
    test_partition= your_database_copy[kkk:j]
    train_partition=your_database_copy[0: kkk]
    train_partition.extend(your_database_copy[j: len(your_database_copy)])
#
    for i in range(0, n_folds):
        print 'Fold '+ str(i+1)
        j=i+each_split
        test_partition= your_database_copy[i:j]
        test_partition_actual=set_database[i:j]
        train_partition=[]
      
        if i==0:
            train_partition=your_database_copy[0: len(your_database_copy)]

        if i >0:
            train_partition.extend(your_database_copy[0: i])
            x=your_database_copy[j: len(your_database_copy)]
           
            train_partition.extend(x)
            
        #train_rows=N.shape(train_partition)[0]
        #train_col=N.shape(train_partition)[1]
        #test_rows=N.shape(test_partition)[0]
        #test_col=N.shape(test_partition)[1]

        predictresult=compute_NaiveBayes(test_partition, train_partition,test_partition_actual)
        #print strr
        #print test_partition
        accuracy1,precision_total1,recall_total1,f1_1 =accuracy(predictresult, test_partition)
        print 'Accuracy: ' + str(accuracy1)
        acc.append(accuracy1)
        precision.append(precision_total1)
        recall.append(recall_total1)
        f1.append(f1_1)
    return acc,precision,recall,f1,predictresult
#
#k_fold=10
#
k_fold = input('Enter the value for k in k_fold cross validation: ')

acc_lst,precision_lst,recall_lst,f1_lst,pred=kfold(k_fold)


print ' '
print ' '
print 'Accuracy for the given Dataset: '+str(sum(acc_lst)/float(len(acc_lst))*100)  
print 'Precision for the given Dataset: '+str(sum(precision_lst)/float(len(precision_lst)))
print 'Recall for the given Dataset: '+str(sum(recall_lst)/float(len(precision_lst)))
print 'F1 measure for the given Dataset: '+str(sum(f1_lst)/float(len(f1_lst)))
 

    
    
    