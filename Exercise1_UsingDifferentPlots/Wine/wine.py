import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('winequality-red.csv', delimiter=';')
numerical_attributes = data.drop(columns=['quality'])

plt.figure(figsize=(12, 8))
numerical_attributes.boxplot(rot=45, fontsize=10)
plt.title('Box Plot of Numerical Attributes in Redwine-Quality Dataset')
plt.ylabel('Attribute Values')
plt.show()
