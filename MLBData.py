import csv
import itertools
import numpy as np
import datetime
import random



def LoadData(fileName, start,end):
    inputData=[]
    outputData=[]    
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in itertools.islice(reader, start,end):
            output=np.longfloat(row[5])
            input = [np.longfloat(x) for x in itertools.islice(row , 5,len(row) ) if x != '']    #len(row)
            inputData.append(input)
            outputData.append(output)
    return inputData,outputData


    
def Dates(fileName,split=False,N=0):
    Dates=[]
    ValidationDates=[]
    TestDate = datetime.time(0, 0, 0)
    i=2
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
	header = reader.next()

        for row in reader:
	
		   if(datetime.datetime.strptime(row[0], '%Y%m%d').date() not in Dates):
	   		Dates.append(datetime.datetime.strptime(row[0], '%Y%m%d').date())

	Dates.sort(reverse=True)

	TestDate=Dates[len(Dates)-1]
	numberTestDates=random.randrange(3,7)
	numberTestDates=7
	i=7
	for testDates in range(numberTestDates):
		#randomDate=random.randrange(len(Dates)-16,len(Dates)-1)
		randomDate=i
		ValidationDates.append(Dates[randomDate])
		Dates.remove(Dates[randomDate])
		i+=1
    
    if(split):
    	if(N=='Dates'):
		Dates=split_list(Dates, wanted_parts=len(Dates))
	else:
		Dates=split_list(Dates, wanted_parts=N)
    

    return Dates,ValidationDates,TestDate


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]

def LoadData_RandTstDates(fileName):
    TrainDates,ValidationDates,TestDate=Dates(fileName)
    inputData=[]
    outputData=[]
    validationInputData=[]
    validationOutputData=[]
    testInputData=[]
    testOutputData=[]
    testDataForCSV=[]
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.next()
	i=2
        for row in reader:
	#for row in itertools.islice(reader, 1,2):

		   if(datetime.datetime.strptime(row[0], '%Y%m%d').date() in TrainDates):
            		output=np.longfloat(row[6])
            		input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row)) if x != '']    #len(row)
            		inputData.append(input)
            		outputData.append(output)
		   if(datetime.datetime.strptime(row[0], '%Y%m%d').date() in ValidationDates):
            		output=np.longfloat(row[6])
            		input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row) ) if x != '']    #len(row)
            		validationInputData.append(input)
            		validationOutputData.append(output)	
		   if(datetime.datetime.strptime(row[0], '%Y%m%d').date() == TestDate):
            		output=np.longfloat(row[6])
            		input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row) ) if x != '']    #len(row)
            		testInputData.append(input)
            		testOutputData.append(output)
			outRow=[str(x) for x in itertools.islice(row , 0,7 ) if x != '']	
			testDataForCSV.append(outRow)

    return inputData,outputData,validationInputData,validationOutputData,testInputData,testOutputData,testDataForCSV
    
 

def LoadData_RandTstDates_NFold(fileName, N):
    TrainDates,ValidationDates,TestDate=Dates(fileName,True,N)
    print TrainDates
    print ValidationDates
    print TestDate

    inputData=[[] for _ in range(len(TrainDates))]
    outputData=[[] for _ in range(len(TrainDates))]
    validationInputData=[]
    validationOutputData=[]
    testInputData=[]
    testOutputData=[]
    testDataForCSV=[]
    with open(fileName, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = reader.next()
	#for row in itertools.islice(reader, 8883,8897):
        for row in reader:
		for group in TrainDates:
			if(datetime.datetime.strptime(row[0], '%Y%m%d').date() in group):
				output=np.longfloat(row[6])
				input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row) ) if x != '']    #len(row)
				index=TrainDates.index(group)
				inputData[index].append(input)
				outputData[index].append(output)
				break;
		
		if(datetime.datetime.strptime(row[0], '%Y%m%d').date() in ValidationDates):
			output=np.longfloat(row[6])
			input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row) ) if x != '']    #len(row)
			validationInputData.append(input)
			validationOutputData.append(output)	
		if(datetime.datetime.strptime(row[0], '%Y%m%d').date() == TestDate):
			output=np.longfloat(row[6])
			input = [np.longfloat(x) for x in itertools.islice(row , 7,len(row) ) if x != '']    #len(row)
			testInputData.append(input)
			testOutputData.append(output)
			outRow=[str(x) for x in itertools.islice(row , 0,7 ) if x != '']	
			testDataForCSV.append(outRow)

    return inputData,outputData,validationInputData,validationOutputData,testInputData,testOutputData,testDataForCSV
