import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('AirQualityUCI.csv', delimiter=';')
data.replace({',': '.'}, regex=True, inplace=True)
numeric_columns = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)',
                   'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',
                   'T', 'RH', 'AH']
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

correlation_matrix = data[numeric_columns].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Air Quality Attributes')
plt.show()
