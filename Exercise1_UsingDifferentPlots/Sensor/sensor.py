import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey
from matplotlib.patches import Patch
import random

# Read the data from the CSV file
import pandas as pd
data = pd.read_csv('Plant_1_Generation_Data.csv')

# Create a Sankey diagram
fig, ax = plt.subplots()
sankey = Sankey(ax=ax, unit=None)
unique_plant_ids = data['PLANT_ID'].unique()
nodes = unique_plant_ids.tolist() + ['DC_POWER', 'AC_POWER', 'DAILY_YIELD', 'TOTAL_YIELD']

# Assign random colors to each plant ID
colors = {plant_id: f'#{random.randint(0, 0xFFFFFF):06x}' for plant_id in unique_plant_ids}

# Initialize legend handles and labels
legend_handles = []

for plant_id in unique_plant_ids:
    sankey.add(flows=[-data[data['PLANT_ID'] == plant_id]['DC_POWER'].sum(),
                      -data[data['PLANT_ID'] == plant_id]['AC_POWER'].sum(),
                      data[data['PLANT_ID'] == plant_id]['DAILY_YIELD'].sum(),
                      data[data['PLANT_ID'] == plant_id]['TOTAL_YIELD'].sum()],
               color=colors[plant_id])
    
    # Create a Patch for the legend
    legend_handles.append(Patch(color=colors[plant_id], label=str(plant_id)))

# Plot the Sankey diagram
diagrams = sankey.finish()
diagrams[0].texts[-1].set_position((-0.5, 0.5))  # Adjust position of the TOTAL_YIELD label

# Add DATE_TIME and PLANT_ID to the plot
plt.title('Sankey Diagram for Plant 1 Generation Data')
plt.xlabel('Date and Time (DATE_TIME)')
plt.ylabel('Plant ID (PLANT_ID)')

# Add legend
plt.legend(handles=legend_handles, title='Plant ID', loc='upper right')

plt.show()
