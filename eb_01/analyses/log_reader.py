from __future__ import division
import csv
import time

def filter_pylog(filename,columnName):
	"""
	Syntax: filter_pylog('string of csv file name w/o extension','data')
	Example: filter_pylog('test2','data')

	This function is specific for PsychoPy log csv. It calls in the converted csv fileself.
	First cleans data with random '' and data types.
	Then filters out specific column data types.
	"""
	filtered_pylog=[]
	columnName=columnName.lower()
	csv_name=str(filename) #+ ".csv"
	with open(csv_name) as csvDataFile:
		pydata=list(csv.reader(csvDataFile))
	#clean out empty ' ' in data
	for i in range(0,len(pydata)):
		try:
			pydata[i][1]=pydata[i][1][:-1].lower()
		except IndexError:
			pass
	#convert from 'string of seconds' to 'float of milliseconds'
	for i in range(0,len(pydata)):
		try:
			pydata[i][0]=round(float(pydata[i][0])*1000,2)
		except ValueError:
			pass
	#filter out data by columnName
	for i in range(0,len(pydata)):
		try:
			if pydata[i][1]==str(columnName):
				filtered_pylog.append(pydata[i])
		except IndexError:
			pass
	return filtered_pylog