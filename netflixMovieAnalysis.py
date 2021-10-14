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

# Import the dataset
netflix_df = pd.read_csv('netflix_titles.csv')
netflix_df.head()


