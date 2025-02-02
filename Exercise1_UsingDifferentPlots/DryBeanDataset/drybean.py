import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel("Dry_Bean_dataset.xlsx")

class_counts = data["Class"].value_counts()

plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Bar Plot of Class Distribution")
plt.xticks(rotation=45)
plt.show()