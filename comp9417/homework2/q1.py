from random import randint
from random import randrange
from random import random
import matplotlib.pyplot as plt
import csv
import copy
import string
import pandas as pd
from collections import Counter
import re
from collections import defaultdict
import time
from graphviz import Graph

class BNetwork:
	def __init__(self,name):
		self.name = name
		self.nodes = {}
	def printnetwork(self):
		for i in self.nodes:
			print('Nodes '+i)
			print('Prob: ')
			print(self.nodes[i].prob)
			print('In nodes: ')
			print([x.name for x in self.nodes[i].in_node])
			print('Out nodes: ')
			print([x.name for x in self.nodes[i].out_node])
class betheClusterGraph:
	def __init__(self):
		self.nodes = {}
		self.edges = []
	def printtree2(self):
		tree = Graph()
		for i in self.nodes:
			tree.node(str(i),str(i))
		for i in self.edges:
			tree.edge(i[0],str(i[1]))
		tree.render()
class joinTree:
	def __init__(self):
		self.nodes = {}
	def printtree(self):
		for i in self.nodes:
			print('Cluster '+str(i)+':')
			print('Prob: ')
			print(self.nodes[i].prob)
			print('neighbours: ')
			for j in self.nodes[i].edge:
				print(str(j)+': '+', '.join([str(x.name) for x in self.nodes[i].edge[j]]))
	def printtree2(self):
		tree = Graph()
		for i in self.nodes:
			tree.node(str(i),str(i))
		for i in self.nodes:
			for j in self.nodes[i].edge:
				for k in self.nodes[i].edge[j]:
					tree.edge(str(i),str(k.name))
		tree.render()
	def addvariable(self,cluster,variable,graph):
		for i in cluster.edge:
			for j in cluster.edge:
				for k in cluster.edge[j]:
					if cluster in k.name:
						nodelist = [graph.nodes[x] for x in cluster.name]
						nodelist.append(graph.nodes[variable])
						newcluster = cluster(nodelist)
						newcluster.edge = cluster.edge
						for l in cluster.edge:
							for m in cluster.edge[l]:
								m.edge[l] = newcluster
						newcluster.msg = cluster.msg
						self.nodes[newcluster.name] = newcluster
						del self.nodes[cluster]
						return
						
		print('No neighbour with variable ' +variable)
		return
		
	def mergecluster(self,cluster1,cluster2,graph):
		tempname = (list(cluster1.name)+list(cluster2.name))
		tempname = sorted(set(tempname),key = tempname.index)
		tempnodelist = [graph.nodes[x] for x in tempname]
		tempnode = cluster(tempnodelist)
		self.nodes[tempnode.name] = tempnode
		print('merging '+str(cluster1.name)+' with '+str(cluster2.name))
		del self.nodes[cluster1.name]	
		del self.nodes[cluster2.name]
		
		for i in self.nodes:
			for j in self.nodes[i].edge:
				if cluster1 in self.nodes[i].edge[j]:
					self.nodes[i].edge[j].remove(cluster1)
					self.nodes[i].edge[j].append(tempnode)
					if j not in tempnode.edge:
						tempnode.edge[j] = [self.nodes[i]]
					else:
						tempnode.edge[j].append(self.nodes[i])
				if cluster2 in self.nodes[i].edge[j]:
					self.nodes[i].edge[j].remove(cluster2)
					self.nodes[i].edge[j].append(tempnode)
					if j not in tempnode.edge:
						tempnode.edge[j] = [self.nodes[i]]
					else:
						tempnode.edge[j].append(self.nodes[i])
		return 
		
	def addcluster(self,cluster1):
		#add cluster to graph
		for i in self.nodes:
			if set(cluster1.name).issubset(self.nodes[i].name):
				if cluster1.name not in self.nodes[i].edge: 
					self.nodes[i].edge[cluster1.name] = []
				self.nodes[i].edge[cluster1.name].append(cluster1)
				cluster1.edge[cluster1.name] = [self.nodes[i]]
				self.nodes[cluster1.name] = cluster1
				return
		print("Cant find eligible cluster to add to")
		return
	
	def deletecluster(self,cluster1):
		deleted = False
		for i in self.nodes:
			if set(cluster1.name).issubset(self.nodes[i].name):
				for j in self.nodes[i].edge:
					if cluster1 in self.nodes[i].edge[j]:
						self.nodes[i].edge[j].remove(cluster1)
						deleted = True
		if deleted:
			del self.nodes[cluster1.name]
			print("Nodes deleted")
			return
		print('No cluster that is superset of '+str(cluster1.name))
		return
	
	def findpath(self, clustername1, clustername2):
		#clustername1 is origin, clustername2 is target
		visited = [self.nodes[clustername1]]
		route = [self.nodes[clustername1]]
		#print('looking for a path from '+ str(clustername1)+' to '+str(clustername2))
		while route:
			#print([x.name for x in route])
			curnode = route.pop()
			if curnode.name == clustername2:
				return True
			for i in curnode.edge:
				for j in curnode.edge[i]:
					if j in visited:
						continue
					route.append(j)
			visited.append(curnode)
		return False
	def querycluster(self,query,evidence):
		result = pd.DataFrame()
		result.iloc[0:0]
		startnode = self.nodes[list(self.nodes.keys())[0]]
		for i in self.nodes:
			if set(query).issubset(i):
				startnode = self.nodes[i]
				#return(self.nodes[i].prob.groupby(query,as_index = False).sum())
				break
		route = []
		temproute = [startnode]
		visited = []
		while temproute:
			curnode = temproute.pop()
			msgfrom = []
			for i in curnode.edge:
				for j in curnode.edge[i]:
					if j in visited:
						continue
					temproute.append(j)
					msgfrom.append(j.name)
			visited.append(curnode)
			route.insert(0,curnode)
			curnode.receive = msgfrom
			#print([(x.name) for x in route])
		route = sorted(set(route),key = route.index)
		visited = []
		for i in route:
			#print(i.tempprob)
			i.receivemsg()
			
			for j in i.edge:
				for k in i.edge[j]:
					if k in visited:
						continue
					tempkey = i.tempprob.columns.tolist()
					tempkey.remove('probability')
					#print(list(set(list(j)+list(set(tempkey).intersection(query)))))
					evidencelist = list(set(tempkey).intersection(evidence.keys()))
					#print('evidence list :' +str(evidencelist))
					
					sendprob = i.tempprob
					for l in evidencelist:
						sendprob = sendprob.loc[sendprob[l] == evidence[l]]
						tempsum = sendprob['probability'].sum()
						sendprob['probability'] = sendprob['probability']/tempsum
						#print(sendprob)
					if i.name not in k.msg:
						if not set(list(set(list(j)+list(set(tempkey).intersection(query))))) & set(sendprob.columns.tolist()):
							empty = pd.DataFrame()
							empty.iloc[0:0]
							k.msg[i.name] = empty
						else:
							#print((list(set(tempkey).intersection(query))))
							#print(list(set(list(j)+list(set(tempkey).intersection(query)))))
							if list(set(tempkey).intersection(query)):
								k.msg[i.name] = (sendprob.groupby(list(set(list(set(list(j)).intersection(tempkey))+list(set(tempkey).intersection(query)))),as_index = False).sum())
							else:
								k.msg[i.name] = (sendprob.groupby(list(set(j).intersection(tempkey)),as_index = False).sum())								

			visited.append(i)
		#print([x.name for x in route])
		#print(route[-1].tempprob)
		final = (route[-1].tempprob.groupby(query,as_index = False).sum())
		tempsum = final['probability'].sum()
		final['probability']= final['probability']/tempsum
		return(final)
class cluster:
	def __init__(self,nodelist):
		self.name = tuple([x.name for x in nodelist])
		temp2 = pd.DataFrame()
		self.edge={}
		self.msg = {}
		self.receive = []
		temp2.iloc[0:0]
		for i in nodelist:
			
			tempset = set(i.prob.columns.tolist())
			tempset.remove('probability')
			if not tempset.issubset(self.name):
				continue
			temp2 = join(temp2,i.prob)
		self.prob = temp2
		self.tempprob = self.prob
		
	def receivemsg(self):
		self.tempprob = self.prob
		for i in self.receive:
			if self.msg[i].empty:
				continue
			self.tempprob = join(self.tempprob,self.msg[i])
		return

class bethecluster:
	def __init__(self,nodelist):
		self.name = tuple([x.name for x in nodelist])
		self.edge=[]
		self.msg = {}
		self.receive = {}
		temp2 = pd.DataFrame()
		temp2.iloc[0:0]
		if len(nodelist) == 1:
			self.prob = temp2
			self.name = nodelist[0].name
			return
		for i in nodelist:
			
			tempset = set(i.prob.columns.tolist())
			tempset.remove('probability')
			if not tempset.issubset(self.name):
				continue
			temp2=join(temp2,i.prob)
		self.prob = temp2
		self.tempprob = self.prob

class Node:
	def __init__(self,name):
		self.name = name
		self.in_node = []
		self.out_node = []
		self.neighbour = []
		self.filledge = set()
		self.variables = []
		self.type = ""
		self.prob = pd.DataFrame()
		self.prob.iloc[0:0]
	def given(self):
		in_node_names = [i.name for i in self.in_node]
		return in_node_names
	def outnodes(self):
		in_node_names = [i.name for i in self.out_node]
		return in_node_names
	def probability(self,probtable):
		self.prob = pd.DataFrame(data= probtable)


def ijgp(clustergraph,query):
	t = 0
	diverge = True
	for edge in clustergraph.edges:
		if 0 not in clustergraph.nodes[edge[0]].receive:
			clustergraph.nodes[edge[0]].receive[0] = {}
		if 0 not in clustergraph.nodes[edge[1]].receive:
			clustergraph.nodes[edge[1]].receive[0] = {}
		if(edge[0] in clustergraph.nodes[edge[1]].prob.columns.tolist()):
			clustergraph.nodes[edge[0]].receive[0][edge[1]] = clustergraph.nodes[edge[1]].prob.groupby(edge[0],as_index = False).sum()
		else:
			clustergraph.nodes[edge[0]].receive[0][edge[1]] = clustergraph.nodes[edge[0]].prob
		clustergraph.nodes[edge[1]].receive[0][edge[0]] = clustergraph.nodes[edge[0]].prob
	while diverge:
		t+=1
		diverge=False
		for edge in clustergraph.edges:
			tempprob = pd.DataFrame()
			tempprob.iloc[0:0]
			if t not in clustergraph.nodes[edge[0]].receive:
				clustergraph.nodes[edge[0]].receive[t] = {}
			if t not in clustergraph.nodes[edge[1]].receive:
				clustergraph.nodes[edge[1]].receive[t] = {}
			
			#print('sending msg from '+str(edge[0])+' to ' +str(edge[1]))
			for i in clustergraph.nodes[edge[0]].receive[t-1]:
				#print('checking received msg from '+str(i))
				if i == edge[1]:
					continue
				if clustergraph.nodes[edge[0]].receive[t-1][i].empty:
					continue
				tempprob = join(tempprob,clustergraph.nodes[edge[0]].receive[t-1][i])
			if not tempprob.empty:
				#print(tempprob)
				tempprob.groupby((edge[0]),as_index = False).sum()
			clustergraph.nodes[edge[1]].receive[t][edge[0]] = tempprob
			
			tempprob = pd.DataFrame()
			tempprob.iloc[0:0]
			#print('sending msg from '+str(edge[1])+' to ' +edge[0])
			for i in clustergraph.nodes[edge[1]].receive[t-1]:
				#print('checking received msg from '+i)
				if i == edge[0]:
					continue
					tempprob = join(tempprob,clustergraph.nodes[edge[1]].receive[t-1][i])
			tempprob = join(tempprob,clustergraph.nodes[edge[1]].prob)
			if edge[0] in tempprob.columns.tolist():
				tempprob = tempprob.groupby((edge[0]),as_index = False).sum()
			else:
				tempprob = pd.DataFrame()
				tempprob.iloc[0:0]
			clustergraph.nodes[edge[0]].receive[t][edge[1]] = tempprob
			if not clustergraph.nodes[edge[0]].receive[t][edge[1]].equals(clustergraph.nodes[edge[0]].receive[t-1][edge[1]]):
				diverge = True
	finalprob = pd.DataFrame()
	finalprob.iloc[0:0]
	for i in clustergraph.nodes:
		if type(i) is tuple:
			continue
		#print(i)
		tempprob = pd.DataFrame()
		tempprob.iloc[0:0]
		for msg in clustergraph.nodes[i].receive[t]:
			tempprob = join(tempprob,clustergraph.nodes[i].receive[t][msg])
		#print(tempprob)
		tempsum = tempprob['probability'].sum()
		tempprob['probability']/= tempsum
		#print(tempprob)
		if i in query:
		#	print('IN QUERY')
			finalprob = join(finalprob,tempprob)
	print(finalprob)
def createbethecluster(graph,order):
	betheclustergraph = betheClusterGraph()
	tempgraph = copy.deepcopy(graph)
	newgraph = []
	for i in order:
		temp = copy.deepcopy(graph.nodes[i].neighbour)
		temp.insert(0,graph.nodes[i])
		newcluster = bethecluster(temp)
		if newcluster.prob.empty:
			continue
		betheclustergraph.nodes[newcluster.name] = newcluster
		remove_node(graph.nodes[i])
	clusterlength = len(betheclustergraph.nodes)
	#print(clusterlength)
	for i in graph.nodes:
		newcluster = bethecluster([graph.nodes[i]])
		for j in betheclustergraph.nodes:
			if newcluster.name in j and type(j) is tuple:
				newcluster.edge.append(betheclustergraph.nodes[j])
				betheclustergraph.nodes[j].edge.append(newcluster)
				betheclustergraph.edges.append([newcluster.name,j])
		betheclustergraph.nodes[newcluster.name] = newcluster
	return betheclustergraph	
	
def createjointree(graph,order):
	jointree = joinTree()
	temptree = []
	tempgraph = copy.deepcopy(graph)
	for i in order:
		temp = copy.deepcopy(graph.nodes[i].neighbour)
		temp.insert(0,graph.nodes[i])
		alreadydone = False
		tempname = []
		newcluster = cluster(temp)
		if newcluster.prob.empty:
			continue
		jointree.nodes[newcluster.name] = newcluster
		temptree.append(newcluster.name)
		remove_node(graph.nodes[i])
	
	for i in range(len(temptree)-1,0,-1):
		#print('Connecting edge from node '+str(temptree[i]))
		temp2 = [i for sub in temptree[i+1:] for i in sub]
		common = (set(temptree[i]) & set(temp2))
		#print('looking for common elements ' +str(common))
		if not common:
			continue
		for j in temptree[i+1:]:
			if common.issubset(j) and not jointree.findpath(temptree[i],j):
				
				if tuple(common) not in jointree.nodes[temptree[i]].edge:
					jointree.nodes[temptree[i]].edge[tuple(common)] = []
				jointree.nodes[temptree[i]].edge[tuple(common)].append(jointree.nodes[j])	
				if tuple(common) not in jointree.nodes[j].edge:
					jointree.nodes[j].edge[tuple(common)] = []
				jointree.nodes[j].edge[tuple(common)].append(jointree.nodes[temptree[i]])
	for i in jointree.nodes.copy():
		if not jointree.nodes[i].edge:
			del jointree.nodes[i]
	return jointree

def join(df1,df2):
	if (df1.empty):
		return df2
	if(df2.empty):
		return df1
	key1 = df1.columns.tolist()
	key2 = df2.columns.tolist()
	similarkey = list(set(key2).intersection(key1))
	(similarkey.remove('probability'))
	if not similarkey:
		df1['key'] = 0
		df2['key'] = 0
		tempprob = pd.merge(df1,df2,on='key')
		del df1['key']
		del df2['key']
		del tempprob['key']
	else:
		tempprob = pd.merge(df1,df2,on=similarkey)
	newval = (tempprob['probability_x']*tempprob['probability_y'])
	del tempprob['probability_x']
	del tempprob['probability_y']
	tempprob.insert(len(tempprob.columns),"probability",newval)	
	return tempprob
	
def min_degree(graph):
	order = []
	tempgraph = copy.deepcopy(graph)
	filledgraph = copy.deepcopy(graph)
	maxdegree = 99999999
	tempnode = ""
	while tempgraph.nodes:
		maxdegree = 99999999
		tempnode = ""
		for i in (tempgraph.nodes):
			degree = len(tempgraph.nodes[i].neighbour)
			if degree < maxdegree:
				maxdegree = degree
				tempnode = i
		order.append(tempnode)
		for j in range(len(tempgraph.nodes[tempnode].neighbour)-1):
			for k in range(j+1,len(tempgraph.nodes[tempnode].neighbour)):
				if tempgraph.nodes[tempnode].neighbour[k] not in tempgraph.nodes[tempnode].neighbour[j].neighbour:
					tempgraph.nodes[tempnode].neighbour[j].neighbour.append(tempgraph.nodes[tempnode].neighbour[k])
					tempgraph.nodes[tempnode].neighbour[k].neighbour.append(tempgraph.nodes[tempnode].neighbour[j])
					indexj = 0
					indexk = 0
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[j].name:
							break
						indexj+=1
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[k].name:
							break
						indexk+=1
					filledgraph.nodes[tempnode].neighbour[indexj].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexk])
					filledgraph.nodes[tempnode].neighbour[indexk].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexj])
		remove_node(tempgraph.nodes[tempnode])
		del tempgraph.nodes[tempnode]
	return order, filledgraph

def min_fill2(graph):
	order = []
	tempgraph = copy.deepcopy(graph)
	filledgraph = copy.deepcopy(graph)
	while tempgraph.nodes:
		maxfill = 99999999
		tempnode = ""
		for i in (tempgraph.nodes):
			filledge = 0
			for j in range(len(tempgraph.nodes[i].neighbour)-1):
				for k in range(j+1,len(tempgraph.nodes[i].neighbour)):
					if tempgraph.nodes[i].neighbour[k] not in tempgraph.nodes[i].neighbour[j].neighbour:
						filledge +=1
			if filledge < maxfill:
				maxfill = filledge
				tempnode = i
		order.append(tempnode)
		
		for j in range(len(tempgraph.nodes[tempnode].neighbour)-1):
			for k in range(j+1,len(tempgraph.nodes[tempnode].neighbour)):
				if tempgraph.nodes[tempnode].neighbour[k] not in tempgraph.nodes[tempnode].neighbour[j].neighbour:
					tempgraph.nodes[tempnode].neighbour[j].neighbour.append(tempgraph.nodes[tempnode].neighbour[k])
					tempgraph.nodes[tempnode].neighbour[k].neighbour.append(tempgraph.nodes[tempnode].neighbour[j])
					indexj = 0
					indexk = 0
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[j].name:
							break
						indexj+=1
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[k].name:
							break
						indexk+=1
					filledgraph.nodes[tempnode].neighbour[indexj].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexk])
					filledgraph.nodes[tempnode].neighbour[indexk].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexj])
		remove_node(tempgraph.nodes[tempnode])
		del tempgraph.nodes[tempnode]
	return order, filledgraph
	
def min_fill(graph):
	order = []
	tempgraph = copy.deepcopy(graph)
	filledgraph = copy.deepcopy(graph)
	while tempgraph.nodes:
		maxfill = 99999999
		maxdegree = 99999999
		tempnode = ""
		for i in (tempgraph.nodes):
			filledge = 0
			degree = len(tempgraph.nodes[i].neighbour)
			for j in range(len(tempgraph.nodes[i].neighbour)-1):
				for k in range(j+1,len(tempgraph.nodes[i].neighbour)):
					if tempgraph.nodes[i].neighbour[k] not in tempgraph.nodes[i].neighbour[j].neighbour:
						filledge +=1
			if filledge < maxfill:
				maxfill = filledge
				tempnode = i
			elif filledge == maxfill:
				if degree < maxdegree:
					maxdegree = degree
					tempnode = i
			if degree < maxdegree:
				maxdegree = degree
		order.append(tempnode)
		
		for j in range(len(tempgraph.nodes[tempnode].neighbour)-1):
			for k in range(j+1,len(tempgraph.nodes[tempnode].neighbour)):
				if tempgraph.nodes[tempnode].neighbour[k] not in tempgraph.nodes[tempnode].neighbour[j].neighbour:
					tempgraph.nodes[tempnode].neighbour[j].neighbour.append(tempgraph.nodes[tempnode].neighbour[k])
					tempgraph.nodes[tempnode].neighbour[k].neighbour.append(tempgraph.nodes[tempnode].neighbour[j])
					indexj = 0
					indexk = 0
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[j].name:
							break
						indexj+=1
					for l in filledgraph.nodes[tempnode].neighbour:
						if l.name == tempgraph.nodes[tempnode].neighbour[k].name:
							break
						indexk+=1
					filledgraph.nodes[tempnode].neighbour[indexj].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexk])
					filledgraph.nodes[tempnode].neighbour[indexk].neighbour.append(filledgraph.nodes[tempnode].neighbour[indexj])
		remove_node(tempgraph.nodes[tempnode])
		del tempgraph.nodes[tempnode]
	return order, filledgraph
	
def variableElimination(graph,query,order,evidence):
	newvariablelist = {}
	for i in graph.nodes:
		newvariablelist[tuple(graph.nodes[i].prob.columns.tolist()[:-1])] = graph.nodes[i].prob
	for i in order:
		temp2 = pd.DataFrame()
		temp2.iloc[0:0]
		for j in newvariablelist.copy():
			if i in j:
				if temp2.empty:
					temp2 = newvariablelist[j]
					del newvariablelist[j]
					continue
				temp2 = join(newvariablelist[j],temp2)
				del newvariablelist[j]
		#for j in newvariablelist:
		#	print('newvariablelist: '+str(j))
		#	print(newvariablelist[j])
		keylist = temp2.columns.tolist()
		keylist.remove('probability')
		if i not in query:
			keylist.remove(i)
		if keylist:
			temp2 = (temp2.groupby(keylist,as_index= False).sum())
		newvariablelist[tuple(sorted(tuple(temp2.columns.tolist()[:-1])))]=(temp2)
	
	tempsum = (temp2['probability'].sum())	
	temp2['probability'] = temp2['probability']/tempsum
	#print(temp2)
	return temp2
	
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
	for i in node.neighbour:
		i.neighbour.remove(node)
	return
	
def prune_nodes(listNodes,bn):
	new = copy.deepcopy(bn)
	pruning = True
	while pruning:
		pruning = False
		for j in new.nodes.copy():
			if j in listNodes:
				continue
			if len(new.nodes[j].out_node) == 0:
				remove_node(new.nodes[j])
				del new.nodes[j]
				pruning = True
	print(len(new.nodes))
	print(len(bn.nodes))
	return new
	
def load_file(filename):
	f = open(filename)
	variable = False
	probability = False
	variablename = ""
	bn = BNetwork(filename)
	line = f.readline()
	while line:
		keywords = line.split()
		if "network" in keywords or keywords[0] == '}':
			line = f.readline()
			variable = False
			probability = False
			continue
		if keywords[0] == "variable":
			variable = True
			bn.nodes[keywords[1]] = Node(keywords[1])
			variablename = keywords[1]
			line = f.readline()
			continue
		if keywords[0] == "probability":
			probability = True
			if '|' in keywords:
				temp = []
				for i in range(keywords.index('|')+1,len(keywords)-2):
					add_edge(bn.nodes[keywords[i].replace(',','')],bn.nodes[keywords[2]])
					temp.append(bn.nodes[keywords[i].replace(',','')])
			variablename = keywords[2]
			line = f.readline()
			tempdict = {}
			
			for i in bn.nodes[variablename].in_node:
				tempdict[i.name] = []
			tempdict[variablename] = []
			tempdict["probability"] = []
			continue
		if probability == True:
			if keywords[0] == 'table':
				for i in range(len(bn.nodes[variablename].variables)):
					tempdict["probability"].append(float(re.findall("\d+\.\d+",keywords[1+i])[0]))
					tempdict[variablename].append(bn.nodes[variablename].variables[i])
				bn.nodes[variablename].probability(tempdict)
			else:
				
				index = 0
				for i in range(len(bn.nodes[variablename].in_node)):
					tempdict[bn.nodes[variablename].in_node[i].name].append(keywords[i].replace('(','').replace(')','').replace(',',''))
					index+=1
				for i in range(len(bn.nodes[variablename].variables)):
					tempdict[variablename].append(bn.nodes[variablename].variables[i])
					tempdict["probability"].append(float(re.findall("\d+\.\d+",keywords[index+i])[0]))
					if(i < len(bn.nodes[variablename].variables)-1):
						for j in range(len(bn.nodes[variablename].in_node)):
							tempdict[bn.nodes[variablename].in_node[j].name].append(keywords[j].replace('(','').replace(')','').replace(',',''))
				bn.nodes[variablename].probability(tempdict)
		if variable == True:
			bn.nodes[variablename].type = keywords[1]
			for i in range(keywords.index(']')+2,len(keywords)-1):
				bn.nodes[variablename].variables.append(keywords[i].replace(',',''))
		
		line = f.readline()
	return bn

def disconnectedge(node1,node2,graph):
	#node 1 is in_node
	remove_edge(node1,node2)
	tempgraph = copy.deepcopy(graph)
	moralgraph(tempgraph)
	lis,newgraph = min_fill(tempgraph)
	tempname = [x.name for x in node2.in_node]
	tempname.append(node2.name)
	newprob = variableElimination(newgraph,tempname,lis,{})
	
	node2.prob = newprob

def connectedge(node1,node2,graph):
	#node 1 is in_node
	add_edge(node1,node2)
	tempgraph = copy.deepcopy(graph)
	moralgraph(tempgraph)
	lis,newgraph = min_fill(tempgraph)
	tempname = [x.name for x in node2.in_node]
	tempname.append(node2.name)
	newprob = variableElimination(newgraph,tempname,lis,{})
	temp = newprob.groupby(node2.name,as_index = False).sum()
	newprob = pd.merge(temp,newprob,on=node2.name)
	new = (newprob['probability_y']/newprob['probability_x'])
	newprob.insert(len(newprob.columns),"probability",new)
	del newprob['probability_x']
	del newprob['probability_y']
	node2.prob = newprob
				
def save_file(bn,filename):
	f = open(filename,'w+')
	for i in bn.nodes:
		f.write("variable "+i+" {")
		f.write("  type "+bn.nodes[i].type+" ["+str(len(bn.nodes[i].variables))+"] { "+", ".join(bn.nodes[i].variables)+" };")
		f.write("}")
	for i in bn.nodes:
		if len(bn.nodes[i].in_node) == 0:
			f.write("probability ( "+i+" ) {")
			f.write("  table "+", ".join(map(str,bn.nodes[i].prob['probability'].values.tolist()))+";")
			f.write("}")
		else:
			names = [j.name for j in bn.nodes[i].in_node ]
			f.write("probability ( "+i+" | "+" ,".join(names)+" ) {")
			#print(bn.nodes[i].prob)
			for j in range(0,bn.nodes[i].prob.shape[0],len(bn.nodes[i].variables)):
				temp = []
				temp2 = []
				for k in range(bn.nodes[i].prob.shape[1]-2):
					temp.append(bn.nodes[i].prob.iloc[j,k])
				tempprob = bn.nodes[i].prob['probability'].values.tolist()
				for k in range(len(bn.nodes[i].variables)):
					temp2.append(str(tempprob[j+k]))
				f.write("  ("+", ".join(temp)+") " +", ".join(temp2))
			f.write("}")

def moralgraph(graph):
	for i in graph.nodes:
		if not graph.nodes[i].in_node:
			continue
			
		for j in range(len(graph.nodes[i].in_node)-1):
			for k in range(j+1,len(graph.nodes[i].in_node)):
				if graph.nodes[i].in_node[k] not in graph.nodes[i].in_node[j].neighbour:
					graph.nodes[i].in_node[k].neighbour.append(graph.nodes[i].in_node[j])
					graph.nodes[i].in_node[j].neighbour.append(graph.nodes[i].in_node[k])
	
temp = load_file("alarm.bif")
#save_file(temp,"new")
#new = prune_nodes(['CombVerMo','InsChange'],temp)
#new = prune_nodes(['GOAL_56','GOAL_57'],temp)
moralgraph(temp)
lis,newgraph = min_fill(temp)
lis2,newgraph2 = min_degree(temp)
lis3,newgraph3 = min_fill2(temp)



jt = createjointree(newgraph,lis)
start = time.time()
#print(jt.querycluster(['GOAL_56','GOAL_57'],{}))
#print(jt.querycluster(['CombVerMo','InsChange'],{}))
end = time.time()
#print('min fill heuristics = '+(str(end-start)))


#jt.printtree2()
#connectedge(temp.nodes['asia'],temp.nodes['smoke'],temp)
#temp.printnetwork()
#print(lis)
#print(lis2)
#print(lis3)
#print([i for i in jt.nodes])
#print(jt.querycluster(['GOAL_56','GOAL_57'],{}))
bcluster = createbethecluster(temp,lis)
bcluster.printtree2()
ijgp(bcluster, ['ARTCO2','BP'])
#print('Merging clusters..')
#jt.deletecluster(('bronc'))
#jt.printtree()

#start = time.time()
#print(variableElimination(temp,['smoke','asia'],lis3,{}))
#end = time.time()
#print('min fill heuristics = '+(str(end-start)))