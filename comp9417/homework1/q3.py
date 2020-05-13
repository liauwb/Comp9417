import networkx as nx
from random import randint
from random import randrange
from random import random
import csv
import copy
import pandas as pd
from collections import Counter
class Node:
	def __init__(self,name):
		self.name = name
		self.in_node = []
		self.out_node = []
		self.neighbour = []
		self.prob = {}
	def given(self):
		in_node_names = [i.name for i in self.in_node]
		return in_node_names
	def outnodes(self):
		in_node_names = [i.name for i in self.out_node]
		return in_node_names
def add_edge(node1,node2):
	node1.out_node.append(node2)
	node2.in_node.append(node1)
	node1.neighbour.append(node2)
	node2.neighbour.append(node1)
	
def remove_edge(node1,node2):
	node1.out_node.remove(node2)
	node2.in_node.remove(node1)
	node1.neighbour.remove(node2)
	node2.neighbour.remove(node1)

def remove_node(node):
	for i in node.in_node:
		i.out_node.remove(node)
	for i in node.out_node:
		i.in_node.remove(node)
	return

def iscyclic(node,path):
	if node in path:
		return True
	path.append(node)
	for i in node.out_node:
		if iscyclic(i,path):
			return True
		path.remove(node)
	return False
	
def generate_dag():
	G = {}
	G['Age'] = Node('Age')
	G['Location'] = Node('Location')
	G['BreastDensity']= Node('BreastDensity')
	G['Size'] = Node('Size')
	G['LymphNodes'] = Node('LymphNodes')
	G['Metastasis']= Node('Metastasis')
	G['BC'] = Node('BC')
	G['Mass'] = Node('Mass')
	G['Shape'] = Node('Shape')
	G['MC'] = Node('MC')
	G['AD'] = Node('AD')
	G['Margin'] = Node('Margin')
	G['SkinRetract'] = Node('SkinRetract')
	G['FibrTissueDev'] = Node('FibrTissueDev')
	G['NippleDischarge']= Node('NippleDischarge')
	G['Spiculation']= Node('Spiculation')
	
	add_edge(G['Age'],G['BC'])
	add_edge(G['Location'],G['BC'])
	add_edge(G['BreastDensity'],G['Mass'])
	add_edge(G['BC'],G['Mass'])
	add_edge(G['BC'],G['Metastasis'])
	add_edge(G['BC'],G['MC'])
	add_edge(G['BC'],G['SkinRetract'])
	add_edge(G['BC'],G['NippleDischarge'])
	add_edge(G['BC'],G['AD'])
	add_edge(G['Mass'],G['Size'])
	add_edge(G['Mass'],G['Shape'])
	add_edge(G['Mass'],G['Margin'])
	add_edge(G['AD'],G['FibrTissueDev'])
	add_edge(G['FibrTissueDev'],G['SkinRetract'])
	add_edge(G['FibrTissueDev'],G['Spiculation'])
	add_edge(G['FibrTissueDev'],G['NippleDischarge'])
	add_edge(G['Spiculation'],G['Margin'])
	add_edge(G['Metastasis'],G['LymphNodes'])
	return G
	
def topological_sort(G,root):
	visited = []
	stack = list(root)
	while stack:
		node = stack.pop()
		visited.append(node)
		while G[node].out_node:
			tempnode = G[node].out_node[0]
			remove_edge(G[node],tempnode)
			if len(tempnode.in_node) == 0:
				stack.append(tempnode.name)
	return visited
	
def sampling(G,file):
	data = pd.read_csv(file)
	roots = [g for g in G if len(G[g].out_node)>0 and len(G[g].in_node)==0]
	order = topological_sort(copy.deepcopy(G),roots)
	
	for i in range(data.shape[0]):
		for j in data:
			if len(G[j].in_node) == 0:
				if data[j][i] not in G[j].prob:
					G[j].prob[data[j][i]]=0
				G[j].prob[data[j][i]]+=1
			else:
				tempkey = []
				for k in G[j].given():
					tempkey.append(data[k][i]) 
				if ",".join(tempkey) not in G[j].prob:
					G[j].prob[",".join(tempkey)] = {}
				if data[j][i] not in G[j].prob[",".join(tempkey)]:
					G[j].prob[",".join(tempkey)][data[j][i]]=0
				G[j].prob[",".join(tempkey)][data[j][i]]+=1
	for i in G:
		if i in roots:
			temp = sum(G[i].prob.values())
			for j in G[i].prob:
				G[i].prob[j] = float(G[i].prob[j])/temp
		else:
			for j in G[i].prob:
				temp = sum(G[i].prob[j].values())
				for k in G[i].prob[j]:
					G[i].prob[j][k] = float(G[i].prob[j][k])/temp
	for i in G:
		print(i)
		print(G[i].prob)
	sampledata =[]
	
	for i in range(1000):
		tempdata = {}
		for j in order:
			temp = random()
			if j in roots:
				for k in G[j].prob:
					temp -= G[j].prob[k]
					if temp <=0.:
						tempdata[j] = k
						break
			else:
				tempgiven = []
				for g in G[j].given():
					tempgiven.append(tempdata[g])
				for k in G[j].prob[",".join(tempgiven)]:
					temp -= G[j].prob[",".join(tempgiven)][k]
					if temp <=0.:
						tempdata[j] = k
						break
		sampledata.append(tempdata)
		
	return sampledata
				
		
				
		
	
	
G = generate_dag()


(sampling(G,"bc 2.csv"))