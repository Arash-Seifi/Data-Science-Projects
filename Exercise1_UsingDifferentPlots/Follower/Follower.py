import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('follow.csv')
G = nx.from_pandas_edgelist(df, 'Follower', 'Followee', create_using=nx.DiGraph())
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=8, font_color='black', font_weight='bold', edge_color='gray', linewidths=0.5)
plt.show()
