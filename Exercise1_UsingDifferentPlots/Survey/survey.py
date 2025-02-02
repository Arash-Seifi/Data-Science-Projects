import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mxmh_survey_results.csv')
genre_counts = df['Fav genre'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(genre_counts, labels=genre_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors)
plt.title('Distribution of Favorite Music Genres')
plt.show()
