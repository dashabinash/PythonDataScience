# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 23:43:22 2018

@author: Abinash Dash
"""
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
"""
   This needs to be analyzed from different material.. Need a detailed clarification
"""


#Missing Data
# This includes demonstration of various methods to deal with missing Datas through the help of Pandas
import numpy as np
import pandas as pd

d={'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]};
print(d);
df=pd.DataFrame(d);
print(df);

print(df.dropna()); # by dropna() method, panadas will drop any row occuring one or more missing values
                    # By default while for dropping row, axis=0;
#    A    B  C
#0  1.0  5.0  1

print(df.dropna(axis=1)); # For dropping column, programmer has to define axis=1
#threshold
#   thresh(argument):int, default None
#   int value: require that many non-NA values
print(df.dropna(thresh=2));
                 # by putting thresh=2 means, programmer wants upto 2 non-NA values
#     A    B  C
#0  1.0  5.0  1
#1  2.0  NaN  2

print(df.dropna(thresh=3));
                # by putting thresh=2 means, programmer wants upto 3 non-NA values
#     A    B  C
#0  1.0  5.0  1

#Column threshold
print(df.dropna(thresh=2,axis=1));
#      A  C
# 0  1.0  1
# 1  2.0  2
# 2  NaN  3

print(df.fillna(value='FILL VALUE')); # This fills value mentioned by programmer in place of NaN values in DataFrame
#             A           B  C
# 0           1           5  1
# 1           2  FILL VALUE  2
# 2  FILL VALUE  FILL VALUE  3

print(df['A'].fillna(value=df['A'].mean()));
# 0    1.0
# 1    2.0
# 2    1.5
# Name: A, dtype: float64

#Group by
# Group by allows you to group together rows based off of a column and perform an aggregate
# function on them

import numpy as np
import pandas as pd

data={'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
       'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
        'Sales':[200,120,340,124,243,350]};
df=pd.DataFrame(data);
print(df);
#  Company   Person  Sales
# 0    GOOG      Sam    200
# 1    GOOG  Charlie    120
# 2    MSFT      Amy    340
# 3    MSFT  Vanessa    124
# 4      FB     Carl    243
# 5      FB    Sarah    350

print(df.groupby('Company'));
#     <pandas.core.groupby.DataFrameGroupBy object at 0x000000B8754FA748>

byComp=df.groupby('Company'); # It returns a groupby object with the address
                              # <pandas.core.groupby.DataFrameGroupBy object at 0x000000B875B3BA20>
print(byComp.mean()); # Pandas will look at the sales column and find the mean of the values
                      # Pandas cannot identify the mean for Person column as these are Sting types and mean()
                      # cannot be found out for Strings
# Company  Sales     
# FB       296.5
# GOOG     160.0
# MSFT     232.0

print(byComp.sum());
#Company    Sales   
#FB         593
#GOOG       320
#MSFT       464

print(byComp.std());
# Company     Sales       
# FB        75.660426
# GOOG      56.568542
# MSFT     152.735065

   # All the above groupby() operations return a DataFrame. So, programmer can apply for Series operation
print(byComp.sum().loc['FB']);
#      Sales    593
#Name: FB, dtype: int64
     
#Now all the above opeartion in one line
print(df.groupby('Company').sum().loc['FB']);
# Sales    593
# Name: FB, dtype: int64

print(df.groupby('Company').count()); # This example is groupby Company by count
# Company    Persons  Sales       
# FB            2      2
# GOOG          2      2
# MSFT          2      2

print(byComp.max());           
# Company    Person  Sales          
# FB         Sarah    350
# GOOG         Sam    200
# MSFT     Vanessa    340

print(byComp.min());
# Company   Persons   Sales             
# FB          Carl    243
# GOOG     Charlie    120
# MSFT         Amy    124

print(byComp.describe());
#Company           Sales      
#FB      count    2.000000
#        mean   296.500000
#        std     75.660426
#       min    243.000000
#        25%    269.750000
#        50%    296.500000
#        75%    323.250000
#        max    350.000000
#GOOG    count    2.000000
#        mean   160.000000
#        std     56.568542
#       min    120.000000
#        25%    140.000000
#        50%    160.000000
#        75%    180.000000
#        max    200.000000
#MSFT    count    2.000000
#        mean   232.000000
#        std    152.735065
#        min    124.000000
#        25%    178.000000
#        50%    232.000000
#        75%    286.000000
#        max    340.000000

print(byComp.describe().transpose()); # It just gives a different format
print(byComp.describe().transpose()['FB']);
#       count   mean        std    min     25%    50%     75%    max
#Sales    2.0  296.5  75.660426  243.0  269.75  296.5  323.25  350.0

# All the above mentioned functions returns a DataFrame


#Merging,Joining and Concatenating
#  This section content demonstrates how to combine multiple DataFrames through various methods

import pandas as pd

df1=pd.DataFrame({'A':['A0','A1','A2','A3'],
                  'B':['B0','B1','B2','B3'],
                   'C':['C0','C1','C2','C3'],
                   'D':['D0','D1','D2','D3']},
                  index=[0,1,2,3]);
                  
df2=pd.DataFrame({'A':['A4','A5','A6','A7'],
                  'B':['B4','B5','B6','B7'],
                   'C':['C4','C5','C6','C7'],
                   'D':['D4','D5','D6','D7']},
                  index=[4,5,6,7]);
                  
df3=pd.DataFrame({'A':['A8','A9','A10','A11'],
                  'B':['B8','B9','B10','B11'],
                   'C':['C8','C9','C10','C11'],
                   'D':['D8','D9','D10','D11']},
                  index=[8,9,10,11]);
            
print(df1);
print(df2);
print(df3);

#Concatenation:- 
#    It usually glues together DataFrames. programmer should keep in mind that dimensions should match along 
#    the axis you are concatenating on. Programmer can use pd.concat and pass a list of DataFrames to
#    concatenate together.
print(pd.concat([df1,df2,df3],axis=0));  # Default axis=0

print(pd.concat([df1,df2,df3],axis=1));  # axis=1 means, if programmer wants to join and merge along with the columns
                                         # or concatenate clong the columns
                                         

left=pd.DataFrame({'key':['k0','k1','k2','k3'],
                    'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']});
                   
right=pd.DataFrame({'key':['k0','k1','k2','k3'],
                    'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']});

print(left);
print(right);

#Merging
#  The merge function allows programmer to merge DataFrames together using a simillar logic as merging SQL Tables 
#  together
print(pd.merge(left,right,how="inner",on="key"));
       # In above command both left and right DataFrame are being merged based on their common key "key" and join "inner"
       # By default all the joining will be "inner"


left=pd.DataFrame({'key1':['k0','k0','k1','k2'],
                   'key2':['k0','k1','k0','k1'],
                    'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']});
                   
right=pd.DataFrame({'key1':['k0','k1','k1','k2'],
                    'key2':['k0','k0','k0','k0'],
                    'C':['C0','C1','C2','C3'],
                     'D':['D0','D1','D2','D3']});
#Below merging example shows how to merge two DataFrames having more than one key
#All the keys have to be passed as a List
print(pd.merge(left,right,how='inner',on=['key1','key2']));
print(pd.merge(left,right,how='outer',on=['key1','key2']));
print(pd.merge(left,right,how='left',on=['key1','key2']));
print(pd.merge(left,right,how='right',on=['key1','key2']));

#Joining
# Joining is a convinient method for combining the columns of two potentially differently-indexed DataFrames into a
# single result DataFrame
#   It is quite simillar to merge operation, here key is actually in index instead of column like in merge operation

jleft=pd.DataFrame({'A':['A0','A1','A2'],
                   'B':['B0','B1','B2']},
                    index=['k0','k1','k2']);
                   

jright=pd.DataFrame({'C':['C0','C2','C3'],
                   'D':['D0','D2','D3']},
                    index=['k0','k2','k3']);

print(jleft.join(jright));
print(jleft.join(jright,how='outer'));
print(jleft.join(jright,how='left'));
print(jleft.join(jright,how='right'));

  
#Operations

import numpy as np
import pandas as pd

df=pd.DataFrame({'col1':[1,2,3,4],
                 'col2':[444,555,666,444],
                 'col3':['abc','def','ghi','xyz']});
print(df);
print(df.head());

#Finding out the unique value in any row or column of a DataFrame
print(df['col2'].unique()); # It returns a numpy array
print(len(df['col2'].unique())); # It returns the length total number of unique elements in that column of DataFrame
print(df['col2'].nunique()); # same as previous command.. instead of len() function nunique() also can be used

#Finding out how many times a unique value occurs for a specific DataFrame
print(df['col2'].value_counts()); # This returns a Series as output
# 444    2
# 555    1
# 666    1
# Name: col2, dtype: int64

#Importance of condistional statement in Pandas
print(df);
print(df[df['col1']>2]); #returns DataFrame
print(df['col1']>2); # returns Series
#Mutiple conditions
print(df[(df['col1']>2) & (df['col2']==444)]); # returns a DataFrame with satisfied conditional output
print(df[(df['col1']==1) & (df['col2']>444)]);
# Empty DataFrame
# Columns: [col1, col2, col3]
# Index: []

#apply() method in Pandas
def timeS2(x):
    return x*2;
    
print(df);
print(df['col1'].sum());
#apply() method
print(df['col1'].apply(timeS2)); # It returns a series
                                 # apply() method calls timeS2() and pass all the columns value as argument into the
                                 # function.
print(df['col3'].apply(len));

#NOTE :
#    apply() method is so powerful if it will be used along with lambda expression

print(df['col2'].apply(lambda x: x*2));

print(df);
print(df.drop(0,axis=0)); # Row removal from a DataFrame. axis=0 for row always
print(df.drop('col1',axis=1)); # Column removal from a DataFrame. axis=1 for column always

print(df.columns);
        # It just gives the details of the columns of a DataFrame.
#        Index(['col1', 'col2', 'col3'], dtype='object')
#          here all the columns are returned in a List and the data type is Object data type

print(df.index);
          # It returns back the index.
          # RangeIndex(start=0, stop=4, step=1)
          # Since it is RangeIndex in the above case, it actually just reports back start,stop and step size of the index

#Sorting the DataFrame
print(df.sort_values('col2')); # This sorts the whole DataFrame based on the values of col2
                               # df.sort_values(by='col2');
                             
print(df.isnull()); # This returns a DataFrame of booleans indicating whether or not the value is null or not

data={'A':['foo','foo','foo','bar','bar','bar'],
       'B':['one','one','two','two','one','one'],
       'C':['x','y','x','y','x','y'],
       'D':[1,3,2,5,4,1]};
dft=pd.DataFrame(data);
print(dft);


"""
   Below concepts have to analyzed from different material.. Not clear description mentioned here
"""
#Pivot table
pivot_dataFrame=df.pivot_table(values='D',index=['A','B'],columns=['C']);
   #Getting KeyError 'A'
   
#Data Input and Output
#   CSV
#   Excel
#   HTML
#   SQL
