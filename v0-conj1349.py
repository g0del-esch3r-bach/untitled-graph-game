from enum import Enum
import math
import networkx as nx # type: ignore
import numpy as np # type: ignore
#import matplotlib.pyplot as plt #type: ignore
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #type: ignore

N = 5

class AgentActions(Enum): # agent can delete or create links
    delete = 0
    create = 1

class NodeLinks(Enum): # each pair of nodes either admits a link or is empty
    empty = 0
    link = 1

class GraphMatrix:
    def __init__(self, rows=N, cols=N):
        self.rows = rows
        self.cols = cols
        self.reset()
    
    def reset(self):
        self.linkpos = 0

    