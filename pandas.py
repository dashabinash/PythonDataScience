# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 23:59:47 2018

@author: Abinash Dash
"""

#Pandas is an open source library built on top of numpy
#It allows for fast analysis, data cleaning and preparation
#It excels in performance and productivity
#It also has built-in visualization features
#It can work with data from a wide variety of sources

"""
  Through out pandas tutorial following things will be focused sequentially
      (i)Series
      (ii)DataFrames
      (iii)Missing Data
      (iv)GroupBy
      (v)merging,Joining and Concatenating
      (vi)Operations
      (vii)Data Input and Output
"""
#SERIES
#It is very simillar to numpy array. Infact it has been built on top of numpy array object
#Difference between numpy array and pandas series is that, series can have axis labels means it can be
#indexed by a label.

import numpy as np
import pandas as pd

labels=['a','b','c'];
my_data=[10,20,30];
arr=np.array(my_data);
d={'a':10,'b':20,'c':30};
print(labels);
print(my_data);
print(arr);
print(d);

print(pd.Series(data=labels));
print(pd.Series(data=my_data));
print(pd.Series(data=arr));
print(pd.Series(data=d)); # While doing Series operation for dictionary, Pandas allows key as indexing and
                          # values as their datas

print(pd.Series(data=my_data,index=labels)); # If the size of both data and index does not match then
                                             # will get "ValueError: Wrong number of items passed 3, placement implies 4"
print(pd.Series(my_data,labels)); #Same as above
print(pd.Series(arr,labels)); #numpy arrays does the same thing that list does

#Panda Series can hold a variety of Object types.
print(pd.Series(data=[sum,print,len]));
                
ser1=pd.Series([1,2,3,4],['USA','Germany','USSR','Japan']);
print(ser1);
ser2=pd.Series([1,2,5,4],['USA','germany','Italy','Japan']);
print(ser2);

print(ser2['USA']);#Always search with indexing basis that means, if in series index is int then search
                   # integer wise and same goes for String also.
print(ser1+ser2); # When it cannot find match, then Pandas will put NaN in place of that.
                  #Output:-
                  #Germany    NaN
                  #Italy      NaN
                  #Japan      8.0
                  #USA        2.0
                  #USSR       NaN
                  #germany    NaN
                  #dtype: float64
#Pandas and numpy will always convert stuff to float to retain all the information possible

#DATAFRAMES
#DataFrames is the main tool while working with pandas
import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101);#seed() means is just to make sure that, programmer gets the same random numbers

df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z']);
print(df);
#      W         X         Y         Z
#A  2.706850  0.628133  0.907969  0.503826
#B  0.651118 -0.319318 -0.848077  0.605965
#C -2.018168  0.740122  0.528813 -0.589001
#D  0.188695 -0.758872 -0.933237  0.955057
#E  0.190794  1.978757  2.605967  0.683509

#Each column is a pandas Series
print(df['W']); 
#      A    2.706850
#      B    0.651118
#      C   -2.018168
#      D    0.188695
#      E    0.190794
#      Name: W, dtype: float64
print(type(df['W'])); # <class 'pandas.core.series.Series'> or df.W
#All DataFrames as bunch of Series which share a index
print(df[['W','Z']]); # get multiple columns

#Note:- Multiple columns returns back a DataFrame
#        Single column returns back a Series

#Creating a new column
df['new']=df['W']+df['Y'];
print(df);

#removing columns and assign it to a new data frame
#What is axis argument in drop() function ?
#Ans:- Axis is just for reference for rows and columns in dataframes by pandas
#      0 for rows 
#      1 for columns
#         You can hypothetically think of x axis in 0 and Y axis in 1.
new_df=df.drop('new',axis=1);
print(new_df); # printing df will results all the coloumns including 'new' also
#            So, we have assign it to a new dataframe and assign the drop() function operation to it.

#But, we can achieve this without assigning it to a new DataFrame like below by adding inplace=True
df.drop('new',axis=1,inplace=True);
print(df);

#Removing row from a data frame
df.drop('E',axis=0,inplace=True);
print(df);

#Returns number of rows and columns in the form of tuple
df.shape; # (4, 4)
          # Where first element is row and second element is column
          
#Selecting ROWS
print(df);
#  There are two ways to get rows from a DataFrame
# FIRST WAY:-
print(df.loc['A']);
# W    2.706850
# X    0.628133
# Y    0.907969
# Z    0.503826

# SECOND  WAY:-
print(df.iloc[0]); # Same output as above
#Not only all the columns but also rows are also pandas Series
#Multiple rows return back DataFrame

#Getting multiple rows from a DataFrame
print(df.loc[['A','D']]);
      
# selecting indiviual value from a DataFrame
print(df.loc['A','W']);
      
#selecting subsets from a DataFrame
print(df.loc[['A','B'],['W','Y']]);
      
#_______________________________________________________________________#
import numpy as np
import pandas as pd

from numpy.random import randn

np.random.seed(101);#seed() means is just to make sure that, programmer gets the same random numbers

df=pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z']);
print(df);
#      W         X         Y         Z
#A  2.706850  0.628133  0.907969  0.503826
#B  0.651118 -0.319318 -0.848077  0.605965
#C -2.018168  0.740122  0.528813 -0.589001
#D  0.188695 -0.758872 -0.933237  0.955057
#E  0.190794  1.978757  2.605967  0.683509

#Conditional Selection
print(df>0);
#       W      X      Y      Z
# A   True   True   True   True
# B   True  False  False   True
# C  False   True   True  False
# D   True  False  False   True
# E   True   True   True   True
booldf=df>0;
print(booldf);
#       W      X      Y      Z
# A   True   True   True   True
# B   True  False  False   True
# C  False   True   True  False
# D   True  False  False   True
# E   True   True   True   True

print(df[booldf]); # Primaryly it returns the result matrix or DataFrame with all satisfied condtional statements results
                   # It returns the number where condiotnal statement is True and returns NaN where 
                   # conditional statement is False
#                             W         X         Y         Z
#                   A  2.706850  0.628133  0.907969  0.503826
#                   B  0.651118       NaN       NaN  0.605965
#                   C      NaN  0.740122  0.528813       NaN
#                   D  0.188695       NaN       NaN  0.955057
#                   E  0.190794  1.978757  2.605967  0.683509

print(df['W']>0);
# A     True
# B     True
# C    False
# D     True
# E     True
# Name: W, dtype: bool
print(df[df['W']>0]); # False satisfied row will be ignored
#                                W         X         Y         Z
#                          A  2.706850  0.628133  0.907969  0.503826
#                          B  0.651118 -0.319318 -0.848077  0.605965
#                          D  0.188695 -0.758872 -0.933237  0.955057
#                          E  0.190794  1.978757  2.605967  0.683509

print(df[df<0]);
#          W         X         Y         Z 
# A       NaN       NaN       NaN        NaN
# B       NaN  -0.319318  -0.848077       NaN
# C  -2.018168       NaN       NaN   -0.589001
# D       NaN  -0.758872   -0.933237       NaN
# E       NaN       NaN       NaN       NaN
print(df[df['Z']<0]);
      
resultdf=df[df['W']>0];
print(resultdf);
print(resultdf['X']);
      
print(df[df['W']>0]['X']); #All the steps metioned in 197,198,199 can be clubbed into one line like this
# A    0.628133
# B   -0.319318
# D   -0.758872
# E    1.978757
       # It returns a pandas series of X column
    
#For multiple columns
print(df[df['W']>0][['X','Y']]); # Pass all the columns name you want to display as List
#          X         Y
# A  0.628133  0.907969
# B -0.319318 -0.848077
# D -0.758872 -0.933237
# E  1.978757  2.605967

#Multiple conditions
print(df[(df['W']>0) & (df['Y']>1)]); # for and operations Pandas does not understand "and" keyword.
                                      # Instead of "and", programmer can use "&"
# Selecting specific columns in one line command after the execution of above statement
print(df[(df['W']>0) & (df['Y']>1)]['X']); # E    1.978757
#                                            Name: X, dtype: float64

print(df[(df['W']>0) | (df['Y']>1)]); # for or operations Pandas does not understand "or" keyword.
                                      # Instead of "or", programmer can use "|"(pipe symbol) 
                                      

#Resetting the index and setting it to something else
print(df);
copy_dataframe=df.copy(); # copy DataFrame from the original DataFrame
print(copy_dataframe);
print(df.reset_index()); # By doing this, index of DataFrame will be set to a column and index starting from
                         # 0 to nth will be the new index
print(df);# the above reset_index() function will not change the original DataFrame. To change that, programmer
          # has to use inplace=True in the reset_index() method
          
newind='CA NY WY OR CO'.split();
print(newind);
df['States']=newind;
print(df);
print(df.set_index('States')); # Setting a new column to index of DataFrame


#MULTI LEVEL INDEX and INDEX HIERCAHY
import numpy as np
import pandas as pd

outside=['G1','G1','G1','G2','G2','G2'];
print(outside);
#     ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']
inside=[1,2,3,1,2,3];
print(inside);
#    [1, 2, 3, 1, 2, 3]
hier_index=list(zip(outside,inside));
print(hier_index);
#    [('G1', 1), ('G1', 2), ('G1', 3), ('G2', 1), ('G2', 2), ('G2', 3)]
hier_index=pd.MultiIndex.from_tuples(hier_index); # did not understabd the command completely
print(hier_index);
#    MultiIndex(levels=[['G1', 'G2'], [1, 2, 3]],
#           labels=[[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])

df=pd.DataFrame(randn(6,2),hier_index,['A','B']); # Did not understand the command completely
print(df);
#             A         B
#G1 1 -0.497104 -0.754070
#   2 -0.943406  0.484752
#   3 -0.116773  1.901755
#G2 1  0.238127  1.996652
#   2 -0.993263  0.196800
#   3 -1.136645  0.000366

#Selecting datas from DataFrame
#   If you want to grab elements based on the outer index.
print(df.loc['G1']); # It returns a subset DataFrame
#          A         B
# 1 -0.497104 -0.754070
# 2 -0.943406  0.484752
# 3 -0.116773  1.901755

#   If you want to grab elements based on the inner index
print(df.loc['G1'].loc[1]); # It returns a Series
# A   -0.497104
# B   -0.754070
# Name: 1, dtype: float64

df.index.names=['Groups','Num'];
print(df);

#NOTE :-
#   (i)While grabing elements from a multilevel index, flow will be always from outer index to inner index


#                   A         B
# Groups Num                    
# G1     1   -0.497104 -0.754070
#        2   -0.943406  0.484752
#        3   -0.116773  1.901755
# G2     1    0.238127  1.996652
#        2   -0.993263  0.196800
#        3   -1.136645  0.000366

#Homework:-
#   Grab 0.196800 from the above DataFrame
print(df.loc['G2'].loc[2]['B']);
      
#Cross section in DataFrame


  










                         



























