import numpy as nmpy
from sklearn import metrics,preprocessing,cross_validation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import sklearn
import sklearn.linear_model as lm
import pandas as pnds


loadData = lambda f: nmpy.genfromtxt(open(f,'r'), delimiter=' ')

def main():

  print "loading data.."
  traindata = list((nmpy.array(pnds.read_table('topicModel_train.tsv')))[:,1:-1])
  testdata = list((nmpy.array(pnds.read_table('topicModel_test.tsv')))[:,1:])
  y = nmpy.array(pnds.read_table('topicModel_train.tsv'))[:,-1]

  print traindata

  rd2 = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

  X_all = traindata + testdata
  lentrain = len(traindata) 
  print(lentrain)
#  print "fitting pipeline"
#  tfv_3gram.fit(X_all)
  #countv.fit(X_all)
#  print "transforming data"
  
#  X_all = tfv_3gram.transform(X_all)
  '''
  X_all = countv.transform(X_all)
  '''

  X = traindata
  X_test = testdata
 # X = X_all[:lentrain]
  #X = X_all
  #X_test = X_all[lentrain:]
  #X_test = X_all
  #print "10 Fold CV Score rd: ", nmpy.mean(cross_validation.cross_val_score(rd, X, y, cv=10, scoring='roc_auc'))
#  print "10 Fold CV Score rd2: ", nmpy.mean(cross_validation.cross_val_score(rd2, X, y, cv=10, scoring='roc_auc'))

  print "training on data"
  #rd.fit(X,y)
  rd2.fit(X,y)
  
  #pred1 = rd.predict_proba(X_test)[:,1]
  #predTrain1 = rd.predict_proba(X)[:,1]
  
  pred2 = rd2.predict_proba(X_test)[:,1]
  predTrain2 = rd2.predict_proba(X)[:,1]

  #pred2 = nmpy.array(pred2)
  #pred2 = nmpy.reshape(pred2,(len(pred2),1))
  #predTrain2 = nmpy.array(predTrain2)
  #predTrain2 = nmpy.reshape(predTrain2,(len(predTrain2),1))
  
  ###combine
  #predTrain = nmpy.insert(predTrain1  ,1,values = predTrain2[:,0],axis = 1 )
  #predTrain[:,1] = nmpy.random.permutation(predTrain[:,1])
  #predTrain[:,0] = nmpy.random.permutation(predTrain[:,0])
  #predTest = nmpy.insert(pred1,1,values=pred2[:,0],axis = 1)
  #print predTrain.shape()
  #print predTest.shape()
  #print predTrain[1]
  #print predTest[2]
  #print "10 Fold CV Train Score: ", nmpy.mean(cross_validation.cross_val_score(rd2, predTrain2, y, cv=10, scoring='roc_auc'))
  
#  rd2.fit(predTrain,y)
#  predFinal = rd2.predict_proba(predTest)[:,1]

  print "added"
  #print pred1[1][1] 
  testfile = pnds.read_csv('test.tsv', sep="\t", na_values=['?'], index_col=1)
  trainfile = pnds.read_csv('train_raw.tsv', sep="\t", na_values=['?'], index_col=1)
  #testfile = testfile
  pred_df = pnds.DataFrame(pred2, index=testfile.index, columns=['label'])
  predTrain_df = pnds.DataFrame(predTrain2, index=trainfile.index, columns=['label'])
  pred_df.to_csv('LR.csv')
  predTrain_df.to_csv('TRAINLR.csv')
  print "submission file created.."

if __name__=="__main__":
  main()
