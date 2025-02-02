import matplotlib.pyplot as plt
import pandas as pd
from math import pi

data = [
    ['Andy Allanson', 293, 66, 1, 30, 29, 14],
    ['Alan Ashby', 315, 81, 7, 24, 38, 39],
    ['Alan Ashby', 315, 81, 7, 24, 38, 39],
    ['Alvin Davis', 479, 130, 18, 66, 72, 76],
    ['Andre Dawson', 496, 141, 20, 65, 78, 37],
    ['Bob Boone', 22, 0, 4, 2, 1, 37],
    ['Doug Baker', 24, 3, 0, 1, 0, 2],
    ['Leon Durham', 484, 127, 10, 66, 67, 7],

]

columns = ['Name', 'AtBat_1986', 'Hits_1986', 'HR_1986', 'Runs_1986', 'RBI_1986', 'Walks_1986']
selected_data = pd.DataFrame(data, columns=columns)

categories = list(selected_data.columns[1:])

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for _, player in selected_data.iterrows():
    values = player[1:].tolist()
    values += values[:1] 
    ax.plot(
        [n / float(len(categories)) * 2 * pi for n in range(len(categories) + 1)],
        values,
        label=player['Name']
    )

ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(0)

ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
ax.set_xticks([n / float(len(categories)) * 2 * pi for n in range(len(categories))])
ax.set_xticklabels(categories)

plt.title('Player Comparison - Radar Chart')
plt.show()
