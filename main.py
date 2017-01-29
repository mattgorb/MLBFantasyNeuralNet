import numpy as np
from NNClass import MLBStatsNeuralNet
import MLBData

def Run(NetworkParameters):
	if(NetworkParameters["NFoldCrossValidation"]):

		TrainingIn,TrainingOut,ValidationIn,ValidationOut,TestIn,TestOut,TestDataForCSV=MLBData.LoadData_RandTstDates_NFold('inputfiles/20160724_1B.csv',NetworkParameters["N"])
		
		for trainingData in TrainingIn:
			index=TrainingIn.index(trainingData)
			Data={'TrainingIn':TrainingIn[index],'TrainingOut':TrainingOut[index],'TestIn':TestIn,'TestOut':TestOut,'ValidationIn':ValidationIn,'ValidationOut':ValidationOut,'TestDataForCSV':TestDataForCSV}
			MLBStatsNeuralNet(Data,NetworkParameters)


	else:
		TrainingIn,TrainingOut,ValidationIn,ValidationOut,TestIn,TestOut,TestDataForCSV=MLBData.LoadData_RandTstDates('inputfiles/20160716_2params.csv')
		
		Data={'TrainingIn':TrainingIn,'TrainingOut':TrainingOut,'TestIn':TestIn,'TestOut':TestOut,'ValidationIn':ValidationIn,'ValidationOut':ValidationOut,'TestDataForCSV':TestDataForCSV}

		MLBStatsNeuralNet(Data,NetworkParameters)

#NetworkParameters examples
"""hiddenDim=5 dropout=True dropoutPercent=.1 learningRate=.01 momentum=0.01 
miniBtch_StochastGradTrain=True miniBtch_StochastGradTrain_Split=30 #for stochastic training set miniBtch_StochastGradTrain="Stochastic"
addInputBias=True biasLayer1TF=True trainingIterations=500 NFoldCrossValidation=True 
N=5 If you want Nfold By dates set N='Dates'
"""


Run({"hiddenDim":3,
"learningRate":.05,
"miniBtch_StochastGradTrain":False,"miniBtch_StochastGradTrain_Split":250,
"trainingIterations":25000,
"momentum":.1,
"dropout":True,"dropoutPercent":0.65,
"biasLayer1TF":True,"addInputBias":True,
"NFoldCrossValidation":True,"N":'Dates',
"shuffle":True
}
)	

