import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('goldstock.csv', parse_dates=['Date'])

df = df.sort_values(by='Date')

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Close'], marker='o', linestyle='-', color='b')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.title('Gold Closing Prices Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
