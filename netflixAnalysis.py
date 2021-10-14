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

# Import the dataset
netflix_df = pd.read_csv('netflix_titles.csv')
netflix_df.head()


#--- Cleaning the data ---#
# We can see from the first 5 rows that some observations
# contain NaN values.
# Looking for missing data
msno.matrix(netflix_df)
plt.show()



