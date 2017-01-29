import numpy as np
import os  
import csv
import math
import sys
import operator
from collections import deque

class MLBStatsNeuralNet:
    layer1_delta=np.array([])
    layer2_delta=np.array([])
    
    weights0MinVal=np.array([])
    weights1MinVal=np.array([])
    weights0MinTest=np.array([])
    weights1MinTest=np.array([])

    runKey=1;
    

    def __init__(self, Data,NetworkParameters):
        self.TrainingIn=np.array(Data['TrainingIn'])
	self.TestDataForCSV=Data['TestDataForCSV']
	
        self.TrainingOut=np.array([Data['TrainingOut']]).T
        self.ValidationIn=np.array(Data['ValidationIn'])
        self.ValidationOut=np.array([Data['ValidationOut']]).T
        self.TestIn=np.array(Data['TestIn'])
        self.TestOut=np.array([Data['TestOut']]).T
        self.hiddenDim=NetworkParameters["hiddenDim"]
        self.minimumValidationError=1
	self.maxRankingPer=0;
	self.TestError=1
	self.lowestTestError=1;
        self.dropoutPercent=NetworkParameters["dropoutPercent"]
        self.dropout=NetworkParameters["dropout"]
        self.learningRate=NetworkParameters["learningRate"]
        self.momentum=NetworkParameters["momentum"]
        self.miniBtch_StochastGradTrain=NetworkParameters["miniBtch_StochastGradTrain"]
        self.miniBtch_StochastGradTrain_Split=NetworkParameters["miniBtch_StochastGradTrain_Split"]
        self.biasLayer1TF=NetworkParameters["biasLayer1TF"]
        self.addInputBias=NetworkParameters["addInputBias"]
	self.trainingIterations=NetworkParameters["trainingIterations"]
	self.shuffle=NetworkParameters["shuffle"]
	self.NFoldCrossValidation=NetworkParameters["NFoldCrossValidation"]
	self.N=NetworkParameters["N"]
        if(self.addInputBias):    
            self.TrainingIn = np.hstack((self.TrainingIn, [[1]] * len (self.TrainingIn) ))
            self.TestIn = np.hstack((self.TestIn, [[1]] * len (self.TestIn) ))
            self.ValidationIn = np.hstack((self.ValidationIn, [[1]] * len (self.ValidationIn) ))
            

        if(self.miniBtch_StochastGradTrain):
            self.TrainingIn_Split=np.array([np.array_split(self.TrainingIn,self.miniBtch_StochastGradTrain_Split)])
            self.TrainingOut_Split=np.array([np.array_split(self.TrainingOut,self.miniBtch_StochastGradTrain_Split)])
	if(self.miniBtch_StochastGradTrain=="Stochastic"):
	    #for stochastic, split=TrainingIn.shape[0]
            self.TrainingIn_Split=np.array([np.array_split(self.TrainingIn,self.TrainingIn.shape[0])])
            self.TrainingOut_Split=np.array([np.array_split(self.TrainingOut,self.TrainingIn.shape[0])])
      
        #initialize weights
        np.random.seed(1)
        self.weights0 = (2*np.random.random((self.TrainingIn.shape[1],self.hiddenDim)) - 1)
        self.weights1 = (2*np.random.random((self.hiddenDim+1 if self.biasLayer1TF else self.hiddenDim,1)) - 1)


        #initialize deltas
        self.layer2_deltaPrev=np.zeros(shape=(self.TrainingOut.shape[0],self.weights1.shape[1]))
        self.layer1_deltaPrev=np.zeros(shape=(self.TrainingIn.shape[0],self.hiddenDim+1 if self.biasLayer1TF else self.hiddenDim))
	

	self.Train()
	self.AddRunToCsv()
	self.CalculateTestErrorFinal()

	print "Max Ranking Percent: "+str(self.maxRankingPer)
	print "Lowest Test Error: " +str(self.lowestTestError)
	print ""

    # sigmoid function
    def nonlin(self,x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))



    # tangent function
    def tan(self,x,deriv=False):
        if(deriv==True):
            return 1.0 - np.tanh(x)**2
        return np.tanh(x)


    def get_last_row(self,csv_filename):
	    with open(csv_filename, 'r') as f:
		for lastrow in csv.reader(f): pass
		return lastrow[0]

    def AddRunToCsv(self):
	global runKey
	if(os.path.isfile('NetworkResults.csv')):
		with open('NetworkResults.csv', 'a') as f:
                	fd = csv.writer(f)
			lastRunKey=self.get_last_row('NetworkResults.csv')
			
			runKey=int(lastRunKey)+1
			fd.writerow([runKey,self.hiddenDim,self.dropout,self.dropoutPercent,self.learningRate,self.momentum,self.miniBtch_StochastGradTrain,self.miniBtch_StochastGradTrain_Split,self.addInputBias,self.biasLayer1TF,self.trainingIterations,self.NFoldCrossValidation,self.N,self.minimumValidationError,self.TestError,self.maxRankingPer])
				
	else:
		with open('NetworkResults.csv', 'wb') as csvfile:
			runKey=1
			fd = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
			fd.writerow(['runKey','hiddenDim','dropout','dropout%','learningRate','momentum','miniBtch','miniBtch_Split','inputBias,biasL1','iterations','NFoldCrVal','N','minimumValidationError','TestError','HighestRank%'])		
			fd.writerow([runKey,self.hiddenDim,self.dropout,self.dropoutPercent,self.learningRate,self.momentum,self.miniBtch_StochastGradTrain,self.miniBtch_StochastGradTrain_Split,self.addInputBias,self.biasLayer1TF,self.trainingIterations,self.NFoldCrossValidation,self.N,self.minimumValidationError,self.TestError,self.maxRankingPer])


    def GetRunKey(self):
	global runKey
	if(os.path.isfile('NetworkResults.csv')):
		with open('NetworkResults.csv', 'a') as f:
                	fd = csv.writer(f)
			lastRunKey=self.get_last_row('NetworkResults.csv')
			return int(lastRunKey)+1	
	else:
		return 1

    def CalculateValidationError(self):
        global weights0MinVal,weights1MinVal
        layer0Test = self.ValidationIn
	
        layer1Test = self.nonlin(np.dot(layer0Test,self.weights0))
        
        if(self.biasLayer1TF):
            biaslayer1Test = np.ones(layer1Test.shape[0])
            layer1Test=np.column_stack((layer1Test,biaslayer1Test))

        layer2Test = self.nonlin(np.dot(layer1Test,self.weights1))

	#print np.average(np.absolute(self.ValidationOut-layer2Test))
        if(np.average(np.absolute(self.ValidationOut-layer2Test))<self.minimumValidationError):
            self.weights0MinVal=self.weights0
            self.weights1MinVal=self.weights1

            self.minimumValidationError=np.average(np.absolute(self.ValidationOut-layer2Test))
	    #self.CalculateTestError()
            print "minimumValidationError: "+str(self.minimumValidationError)
	    self.CalculateTestError()



   








	    
            
    def CalculateTestErrorFinal(self):
        global weights0MinTest,weights1MinTest, runKey
        layer0Test = self.TestIn
	
        layer1Test = self.nonlin(np.dot(layer0Test,self.weights0MinTest))
        
        if(self.biasLayer1TF):
            biaslayer1Test = np.ones(layer1Test.shape[0])
            layer1Test=np.column_stack((layer1Test,biaslayer1Test))

        layer2Test = self.nonlin(np.dot(layer1Test,self.weights1MinTest))
	for row in self.TestDataForCSV:
		index=self.TestDataForCSV.index(row)
		row.append(layer2Test[index][0])
	#self.WriteTestDataToCSV()

	for r in self.TestDataForCSV:
		r[6]=float(r[6])
		r[7]=float(r[7])
	    
    	self.TestDataForCSV_pointsSorted=sorted(self.TestDataForCSV, key=operator.itemgetter(2,6),reverse=True)

    	pos=self.TestDataForCSV_pointsSorted[0][2]
	i=1
    	for r in self.TestDataForCSV_pointsSorted:
		if(r[2]!=pos):
	    		i=1
	    		pos=r[2]
	    		r.append(i)
			i+= 1
		else:
	    		r.append(i)
	    		i+= 1
    
    	self.TestDataForCSV_learnedSorted=sorted(self.TestDataForCSV_pointsSorted, key=operator.itemgetter(2,7),reverse=True)
    	pos=self.TestDataForCSV_learnedSorted[0][2]
    	i=1
    	for r in self.TestDataForCSV_learnedSorted:
		if(r[2]!=pos):
	    		i=1
	    		pos=r[2]
	    		r.append(i)
			i+= 1
		else:
			
	    		r.append(i)
	    		i+= 1
        total=0
	totalTrue=0
    	for r in self.TestDataForCSV_learnedSorted:       
		if(abs(int(r[len(r)-1])-int(r[len(r)-2]))<=3):
			r.append(True)
			totalTrue+=1
	    		total+=1
		else:
			r.append(False)
	    		total+=1

    	for r in self.TestDataForCSV:
		
		r[6]=str(r[6])
		r[7]=str(r[7])
		r[8]=str(r[8])
		r[9]=str(r[9])
		r[10]=str(r[10])
	


	filename="Results/"+str(self.TestDataForCSV[0][0])+"_"+str(runKey)+".csv"
	np.savetxt(filename, self.TestDataForCSV, delimiter=',',fmt='%s', header='date,plyr,pos,id,team,pts,ptsDist,learned,pointsRank,learnedRank,rankCompare')

	for r in self.TestDataForCSV:
		r.pop()
		r.pop()
		r.pop()
		r.pop()
		
	        
            
    def CalculateTestError(self):
        global weights0MinVal,weights1MinVal,lowestTestError,weights0MinTest,weights1MinTest
        layer0Test = self.TestIn

        layer1Test = self.nonlin(np.dot(layer0Test,self.weights0MinVal))
        
        if(self.biasLayer1TF):
            biaslayer1Test = np.ones(layer1Test.shape[0])
            layer1Test=np.column_stack((layer1Test,biaslayer1Test))

        layer2Test = self.nonlin(np.dot(layer1Test,self.weights1MinVal))
        self.TestError=np.average(np.absolute(self.TestOut-layer2Test))


	if(self.TestError<self.lowestTestError):
		self.lowestTestError=self.TestError
		print "New low test error: " +str(self.lowestTestError)
	

	for row in self.TestDataForCSV:
			index=self.TestDataForCSV.index(row)
			row.append(layer2Test[index][0])		

	for r in self.TestDataForCSV:
		r[6]=float(r[6])
		r[7]=float(r[7])
	    
    	self.TestDataForCSV_pointsSorted=sorted(self.TestDataForCSV, key=operator.itemgetter(2,6),reverse=True)

    	pos=self.TestDataForCSV_pointsSorted[0][2]
	i=1
    	for r in self.TestDataForCSV_pointsSorted:
		if(r[2]!=pos):
	    		i=1
	    		pos=r[2]
	    		r.append(i)
			i+= 1
		else:
	    		r.append(i)
	    		i+= 1
    
    	self.TestDataForCSV_learnedSorted=sorted(self.TestDataForCSV_pointsSorted, key=operator.itemgetter(2,7),reverse=True)
    	pos=self.TestDataForCSV_learnedSorted[0][2]
    	i=1
    	for r in self.TestDataForCSV_learnedSorted:
		if(r[2]!=pos):
	    		i=1
	    		pos=r[2]
	    		r.append(i)
			i+= 1
		else:
			
	    		r.append(i)
	    		i+= 1
        total=0
	totalTrue=0
    	for r in self.TestDataForCSV_learnedSorted:       
		if(abs(int(r[len(r)-1])-int(r[len(r)-2]))<=3):
			totalTrue+=1
	    		total+=1
		else:
	    		total+=1

	for r in self.TestDataForCSV:
		r.pop()
		r.pop()
		r.pop()
	
	
	percentCorrect=float(float(totalTrue)/float(total))
	if(percentCorrect>self.maxRankingPer):
		self.maxRankingPer=percentCorrect
		
	        self.weights0MinTest=self.weights0MinVal
            	self.weights1MinTest=self.weights1MinVal

		print "New high ranking %: " +str(self.maxRankingPer)
	




    def MiniBatchOrStochasticGradientDescent(self):
        layer1_delta_accum=[]
        layer2_delta_accum=[]

        for iter in xrange(self.TrainingIn_Split.shape[1]):
            layer0 = self.TrainingIn_Split[0][iter]
            layer1 = self.nonlin(np.dot(layer0,self.weights0))
        
            if(self.biasLayer1TF):
                self.biaslayer1 = np.ones(layer1.shape[0])
                layer1=np.column_stack((layer1,self.biaslayer1))
        
            layer2 = self.nonlin(np.dot(layer1,self.weights1))
            layer2_error = self.TrainingOut_Split[0][iter] - layer2
            layer2_delta = layer2_error*self.nonlin(layer2,deriv=True)*self.learningRate
            layer1_error = layer2_delta.dot(self.weights1.T)
            layer1_delta = layer1_error * self.nonlin(layer1,deriv=True)*self.learningRate
            layer2_delta_accum.extend(layer2_delta.tolist())
            layer1_delta_accum.extend(layer1_delta.tolist())    

        layer2_delta_accum=np.array(layer2_delta_accum)+self.momentum*self.layer2_deltaPrev
        layer1_delta_accum=np.array(layer1_delta_accum)+self.momentum*self.layer1_deltaPrev
        return layer1_delta_accum, layer2_delta_accum

    def Train(self):
        global weights0,weights1,layer1_deltaPrev,layer2_deltaPrev
        for iter in xrange(self.trainingIterations):
	     
            layer0 = self.TrainingIn
	    if(self.shuffle):
	    	np.random.shuffle(layer0)

            layer1 = self.nonlin(np.dot(layer0,self.weights0))

            if(self.biasLayer1TF):
                self.biaslayer1 = np.ones(layer1.shape[0])
                layer1=np.column_stack((layer1,self.biaslayer1))
            if(self.dropout):
                layer1 *= np.random.binomial([np.ones((layer1.shape[0],layer1.shape[1]))],1-self.dropoutPercent)[0] * (1.0/(1-self.dropoutPercent))
            
            layer2 = self.nonlin(np.dot(layer1,self.weights1))
            
            if(self.miniBtch_StochastGradTrain):
                self.layer1_delta,self.layer2_delta=self.MiniBatchOrStochasticGradientDescent()
                self.layer2_deltaPrev=self.layer2_delta
                self.layer1_deltaPrev=self.layer1_delta
            else:

                layer2_error = self.TrainingOut - layer2
                self.layer2_delta = layer2_error*self.nonlin(layer2,deriv=True)*self.learningRate+self.momentum*self.layer2_deltaPrev
                self.layer2_deltaPrev=self.layer2_delta
		
                layer1_error = self.layer2_delta.dot(self.weights1.T)
                self.layer1_delta = layer1_error * self.nonlin(layer1,deriv=True)*self.learningRate+self.momentum*self.layer1_deltaPrev
                self.layer1_deltaPrev=self.layer1_delta
    
            self.weights1 += layer1.T.dot(self.layer2_delta)
            if(self.biasLayer1TF):
                self.weights0 += layer0.T.dot(self.layer1_delta.T[:-1].T)
            else:
                self.weights0 += layer0.T.dot(self.layer1_delta)
            
    
            if (iter% 20) == 0:
                self.CalculateValidationError()
		
                if(self.minimumValidationError<0.01):
                    break


2
