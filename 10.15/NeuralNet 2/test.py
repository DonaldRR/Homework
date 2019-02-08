# A Python program for Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix representation of the graph

import sys # Library for INT_MAX

class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
					for row in range(vertices)]

    def printMST(self, parent):
        print("Edge \t Weight")
        for i in range(1,self.V):
            print(parent[i],"-",i,"\t",self.graph[i][ parent[i] ])

    def minKey(self, key, mstSet):
        min = 100

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    def primMST(self):
        key = [100] * self.V
        parent = [None] * self.V # Array to store constructed MST
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            u = self.minKey(key, mstSet)
            mstSet[u] = True
            print("u:",u)
            print("mstSet:",mstSet)
            print("key:",key)

            for v in range(self.V):

                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
            print("new key:",key)
        self.printMST(parent)

g = Graph(5)
g.graph = [ [0, 2, 0, 6, 0],
			[2, 0, 3, 8, 5],
			[0, 3, 0, 0, 7],
			[6, 8, 0, 0, 9],
			[0, 5, 7, 9, 0]]

g.primMST();

# Contributed by Divyanshu Mehta

