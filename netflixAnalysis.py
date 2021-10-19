'''
Netflix Movie Titles Analysis
Date: 14/10/2021
By Ayah Kamal
'''

# Include libraries and packages to be used
import pandas as pd
import numpy as np
import seaborn as sns
from plotnine import ggplot
import matplotlib.pyplot as plt
import missingno as msno

#--- Import the dataset ---#
netflix_df = pd.read_csv('netflix_titles.csv')
netflix_df.head()

netflix_df.shape # 12 columns, 8807 rows

#--- Cleaning the data ---#
# We can see from the first 5 rows that some observations
# contain NaN values.
# Looking for missing data
msno.matrix(netflix_df)
plt.show()

print('\nColumns with missing value:') 
print(netflix_df.isnull().any())

(netflix_df.isnull().mean()*100).sort_values(ascending=False)[:6]

netflix_df.director.fillna("No Director", inplace=True)
netflix_df.cast.fillna("No Cast", inplace=True)
netflix_df.country.fillna("Country Unavailable", inplace=True)
netflix_df.dropna(subset=["date_added", "rating", "duration"], inplace=True)

#--- EDA and Visualization ---#
plt.figure(figsize=(12,6))
plt.title("Percentage of Netflix Titles as either Movies or TV Shows")
plt.pie(netflix_df.type.value_counts(),explode=(0.01,0.01),labels=netflix_df.type.value_counts().index, colors=['#b1a7a6',"#a4161a"],autopct="%1.2f%%")
plt.show()

