import sys
import re
topicsTrain = dict()
topicsTest = dict()
if __name__ == '__main__':
	infile = file('files/tutorial_composition.txt',"r")
	outfile1 = file('topicModelling/trainConf.csv',"w")
	outfile2 = file('topicModelling/testConf.csv',"w")
	train = re.compile('train[0-9]+')
	test = re.compile('test[0-9]+')
	infile.next()
	for line in infile:
		line = line.strip()
		tokens = line.split()
		findTrain = train.search(tokens[1])
		if (findTrain):
				
			find = findTrain.group(0)
			find = find[5:]
			find = int(find)
			topicsTrain[find] = dict()
			#print len(tokens)
			for topicNo in range(2,len(tokens),2):
				topicsTrain[find][int(tokens[topicNo])] = tokens[topicNo +1]

		findTest = test.search(tokens[1])
		if (findTest):
				
			find = findTest.group(0)
			find = find[4:]
			#print (find)
			find = int(find)
			topicsTest[find] = dict()
			for topicNo in range(2,len(tokens),2):
				topicsTest[find][int(tokens[topicNo])] = tokens[topicNo +1]

	for pop in topicsTrain:
		text = str(pop)
		#print topicsTrain[pop]
		for topic in range(0,12):
			text = text + ' ' + str(topicsTrain[pop][topic])
		outfile1.write(text)
		outfile1.write("\n")
	for pop in topicsTest:
		text = str(pop)
		for topic in range(0,12):
			text = text + ' ' + str(topicsTest[pop][topic])
		outfile2.write(text)
		#outfile2.write(str(pop)+','+str(topicsTest[pop]))
		#print str(pop)+','+str(topicsTest[pop])
		outfile2.write("\n")
		#print '2'


















