import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('car_prices.csv')
df_first_half = df.iloc[:len(df)//9]

plt.scatter(
    df_first_half['year'],
    df_first_half['sellingprice'],
    s=df_first_half['odometer'] / 800, 
    c=df_first_half['odometer'],        
    alpha=0.7                            
)

plt.title('Bubble Plot of Car Prices (First Half of Data)')
plt.xlabel('Year')
plt.ylabel('Selling Price')
plt.colorbar(label='Odometer')
plt.grid(True)
plt.show()
