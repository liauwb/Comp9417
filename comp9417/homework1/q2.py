import networkx as nx
from random import randint
from random import randrange
import matplotlib.pyplot as plt
import csv
import pandas as pd
from collections import Counter

def create_dag():
	G = nx.DiGraph()
	G.add_node('Age')
	G.add_node('Location')
	G.add_node('BreastDensity')
	G.add_node('Size')
	G.add_node('LymphNodes')
	G.add_node('Metastasis')
	G.add_node('BC')
	G.add_node('Mass')
	G.add_node('Shape')
	G.add_node('MC')
	G.add_node('AD')
	G.add_node('Margin')
	G.add_node('SkinRetract')
	G.add_node('FibrTissueDev')
	G.add_node('NippleDischarge')
	G.add_node('Spiculation')
	
	G.add_edge('Age','BC')
	G.add_edge('Location','BC')
	G.add_edge('BreastDensity','Mass')
	G.add_edge('BC','Mass')
	G.add_edge('BC','Metastasis')
	G.add_edge('BC','MC')
	G.add_edge('BC','SkinRetract')
	G.add_edge('BC','NippleDischarge')
	G.add_edge('BC','AD')
	G.add_edge('Mass','Size')
	G.add_edge('Mass','Shape')
	G.add_edge('Mass','Margin')
	G.add_edge('AD','FibrTissueDev')
	G.add_edge('FibrTissueDev','SkinRetract')
	G.add_edge('FibrTissueDev','Spiculation')
	G.add_edge('FibrTissueDev','NippleDischarge')
	G.add_edge('Spiculation','Margin')
	G.add_edge('Metastasis','LymphNodes')
	
	return G
def d_separation(G,file):
	#print(nx.adjacency_matrix(G).todense())
	data = pd.read_csv(file)
	probTable = {}
	outcomeSpace = {}
	#print(data)
	for i in G.nodes:
		if i != 'None':
			probTable[i] = Counter(list(data[i]))
			probTable[i] = dict(probTable[i])
			for j in probTable[i]:
				probTable[i][j] = float(probTable[i][j])/20000
			outcomeSpace[i] = tuple(probTable[i].keys())
	print(probTable)
	print(outcomeSpace)
	#nx.draw(G,with_labels= True)
	#plt.show()
	
	
	
	
G = create_dag()

print(d_separation(G,"bc 2.csv"))