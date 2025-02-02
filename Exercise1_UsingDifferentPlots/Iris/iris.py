import pandas as pd
import matplotlib.pyplot as plt

iris_data = pd.read_csv('iris.data', header=None)
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

setosa = iris_data[iris_data['class'] == 'Iris-setosa']
versicolor = iris_data[iris_data['class'] == 'Iris-versicolor']
virginica = iris_data[iris_data['class'] == 'Iris-virginica']

plt.scatter(setosa['sepal_length'], setosa['sepal_width'], c='r', label='Iris Setosa')
plt.scatter(versicolor['sepal_length'], versicolor['sepal_width'], c='g', label='Iris Versicolor')
plt.scatter(virginica['sepal_length'], virginica['sepal_width'], c='b', label='Iris Virginica')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Sepal Length vs. Width')

plt.legend()
plt.show()