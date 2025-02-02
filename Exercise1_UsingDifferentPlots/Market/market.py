import pandas as pd
import matplotlib.pyplot as plt
import squarify

file_path = 'Amazon Sale Report.csv'
df = pd.read_csv(file_path)

tree_map_data = df[['Category', 'Size', 'Amount']]
tree_map_data_grouped = tree_map_data.groupby(['Category', 'Size']).sum().reset_index()
tree_map_data_grouped.set_index(['Category', 'Size'], inplace=True)

plt.figure(figsize=(10, 8))
squarify.plot(
    sizes=tree_map_data_grouped['Amount'],
    label=tree_map_data_grouped.index,
    alpha=0.7,
    text_kwargs={'fontsize': 8} 
)
plt.title('Gold Market Sales Tree Map')
plt.axis('off')  
plt.show()
