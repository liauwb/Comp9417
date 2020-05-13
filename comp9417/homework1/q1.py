import networkx as nx
from random import randint
from random import randrange
import matplotlib.pyplot as plt

class Node:
	def __init__(self,name):
		self.name = name
		self.in_node = []
		self.out_node = []
		self.neighbour = []
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
		i.node.neighbour.remove(node)
	return


def random_dag(nodes, edges):
	G = {}
	for i in range(nodes):
		G[chr(97+i)] = Node(chr(97+i))
	while edges > 0:
		a = chr(97+randint(0,nodes-1))
		b=a
		while b==a:
			b = chr(97+randint(0,nodes-1))
		add_edge(G[a],G[b])
		edges -=1
	return G
def find_path(x,y,G):
	visited = []
	stack = [x]
	while stack:
		node = stack.pop()
		if node not in visited:
			if node== y:
				return True
			visited.append(node)
			for i in G[node].neighbour:
				stack.append(i.name)
	return False
	
def d_separation(G,X,Y,Z):
	leafs = [g for g in G if len(G[g].out_node)==0 and len(G[g].in_node)>=1]
	print(leafs)
	
	for x in X:
		if x in leafs:
			leafs.remove(x)
	for y in Y:
		if y in leafs:
			leafs.remove(y)
	for z in Z:
		if z in leafs:
			leafs.remove(z)
	for i in leafs:
		remove_node(G[i])
	for z in Z:
		temp = G[z].out_node
		for i in temp:
			remove_edge(G[z],i)
	for x in X:
		for y in Y:
			if find_path(x,y,G):
				return False
	return True
	
	
	
G = random_dag(16,18)


X = [chr(randrange(97,97+15))]
Y = [chr(randrange(97,97+15))]
Z = [chr(randrange(97,97+15))]
print(X)
print(Y)
print(Z)
print(d_separation(G,X,Y,Z))