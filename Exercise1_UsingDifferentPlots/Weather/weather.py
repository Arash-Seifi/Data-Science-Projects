import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('austin_weather.csv')

attributes = ['TempHighF', 'TempAvgF', 'TempLowF', 'DewPointHighF', 'DewPointAvgF', 'DewPointLowF', 'HumidityHighPercent',
              'HumidityAvgPercent', 'HumidityLowPercent', 'SeaLevelPressureHighInches', 'SeaLevelPressureAvgInches',
              'SeaLevelPressureLowInches', 'VisibilityHighMiles', 'VisibilityAvgMiles', 'VisibilityLowMiles', 'WindHighMPH',
              'WindAvgMPH', 'WindGustMPH']

half_data = data.iloc[:len(data)//20]

plt.figure(figsize=(8, 6))
pd.plotting.parallel_coordinates(half_data[attributes], 'WindGustMPH')

plt.legend(loc='upper right', fontsize='small')
plt.xticks(fontsize=5, rotation=45)
plt.yticks(fontsize=8)
plt.xlabel('Attributes', fontsize=10)
plt.ylabel('Attribute Values', fontsize=10)
plt.show()