# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 15:53:09 2018

@author: Abinash Dash
"""
#This tutorial demonstrates about the knowledge on Python
#Numpy is a Linear Algebra Library for Python, the reason it is so important for Data Science with Python
#is that almost all of the libraries in the PyData EcoSystem rely on  Numpy as one of their main building 
#blocks
#It is so fast, as it has bindings to C libraries.
#numpy arrays essentially come in two flavours: vectors and matrices
#Vectors are strictly 1-d arrays and matrices are 2-d (but programmer should note a matrix can still have
#                                                      only one row or one coloumn)
#


my_list=[1,2,3];
print(my_list);

import numpy as np
list_arr=np.array(my_list); # Using numpy normal python list can be casted to an array
print(list_arr) # [1 2 3]
print(type(list_arr));  # <class 'numpy.ndarray'>

my_mat=[[1,2,3],[4,5,6],[7,8,9]];
print(my_mat);
mat_arr=np.array(my_mat); # Will return two dimensional array
print(mat_arr); #[[1 2 3]
 #                [4 5 6]
 #                [7 8 9]]
 
#
arr=np.arange(0,11);
print(arr); #returns a 1-D array. Simillar to range() function of python
step_arr=np.arange(0,11,2);
print(step_arr);

print(np.zeros(3));
#Syntax-np.zeros(i,j)
#       Where i-rows 
#             j-columns
print(np.zeros((5,5)));

print(np.ones(3));
print(np.ones((3,5)));

#linspace array
#There is very slight difference between linspace() and arange() function
#Both takes equal number of arguments
#But arange() takes third argument as step size in function where as linspace() takes third argument
#as the number of points programmer wants with even distribution.
print(np.linspace(0,5)); # If third argument is not given then what is the default third argument ?
print(len(np.linspace(0,5)));
print(np.linspace(0,5,10));
print(np.linspace(0,5,100));

#Identity matrix using numpy
#What is Identity matrix ?
#Ans:- This is a 2-D matrix which is having equal number of rows and columns. Moreover the diagonal 
#      elements of matrix is either all 0s or 1s
print(np.eye(4));
print(np.eye(1));
print(np.eye(0));

print(np.random.rand(5)); #rand() function is going to create an array of the given shape programmer
                          #pass in and it's going to populate with random samples from a uniform
                          #distribution over 0 to 1.
print(np.random.rand(1));      
print(np.random.rand(5,5));

#If programmer wants to return a sample or many samples from the standard normal distribution or
#Gaussian distribution instead of using rand(), randn() can be used.
print(np.random.randn(0));
print(np.random.randn(1));
print(np.random.randn(5));
print(np.random.randn(4,4));


#Usages of randint()
#randint() function three arguments 
#                -> lowest(inclusive)
#                -> highest(exclusive)
#                -> number of random integers programmer wants to display
#This function returns random integers from a low to a high range
print(np.random.randint(0,100));
print(np.random.randint(0,100,10));

arr=np.arange(25);
ranarr=np.random.randint(0,50,10);
print(arr);
print(ranarr);

#Reshape the array
#Means return a new array containing the same data in a new shape
print(arr.reshape(5,5));
#reshape() function throws error, if all the cells will not filled after reshaping

print(ranarr.max()); # returns the max element
print(ranarr.min()); # returns the min element

print(ranarr.argmax()); # returns the index position of max element
print(ranarr.argmin()); # returns the index position of min element

print(arr.shape); # shape() function returns the shape of a arr/vector
                  # O/P- (25,) means arr is an 1-D array
arr=arr.reshape(5,5); # Now returns a 2-D array 
print(arr); 
print(arr.shape); # O/P-(5,5) means arr is 2-D array after reshaping

#Data type of an array
print(arr.dtype); # O/P- int32
print(ranarr.dtype); # O/P- int32


#for random integer other method
from numpy.random import randint

print(randint(0,10));


#Numpy indexing and selection --> Means selecting or picking a subgroup of elements from a group
#                                 of elements in numpy

import numpy as np
select_arr=np.arange(0,11);
print(select_arr);
copy_arr=select_arr.copy(); # copying the array
print(len(select_arr));
print(select_arr[8]);
print(select_arr[1:5]); # includes start and excludes end
print(select_arr[2:]);
print(select_arr[:6]);
select_arr[5:]=100; # Broadcast that value 100 to those first five digits in whole array
print(select_arr);
#sliciing of array and create one new array with the sliced elements
slicing_of_arr=select_arr[1:5];
print(slicing_of_arr);
print(slicing_of_arr[:]);
slicing_of_arr[:]=500;
print(slicing_of_arr);
print(select_arr); # O/P- [  0 500 500 500 500 100 100 100 100 100 100]
                   # even after slicing and broadcasting a new value and then assign it to a new array
                   # it will affect the original array
                   #Python does this behavior to avoid memory issue. To Avoid that, create a copy of your original array
print(copy_arr);
bool_arr=copy_arr > 5;
print(bool_arr);
print(copy_arr[bool_arr]); # It returns elements of the array where condition is True
print(copy_arr[copy_arr>5]);  # same as previous command


#create a 2D array
import numpy as np
arr_2d=np.array([[5,10,15],[20,25,30],[35,40,45]]);
copy_arr_2d=arr_2d.copy();
print(arr_2d);
print(arr_2d[0][0]); # print the exact cell value.. ie first bracket takes row and 2nd bracket takes column
print(arr_2d[0]);#prints first row
#How to grab first column

print(arr_2d[2,1]); # Alternate of arr_2d[2][1]
#Slicing subarray from the original array
print(arr_2d[1:,1:]);
print(arr_2d[1:,0:]);
arr_2d[1:,0:]=100;# Broadcasting a new value in 2D array
print(arr_2d);
slicing_of_2d_arr=arr_2d[0];
print(slicing_of_2d_arr);
slicing_of_2d_arr[:]=50;
print(slicing_of_2d_arr);
print(arr_2d);
print(copy_arr_2d);



#Assignment :-
#      Create an array of 50 elements in 5X10 structure and do the basic operations in the array


#Numpy operations
# This will be discussed in three ways:-
#             Array with Array
#             Array with scalars
#             Universal Array Functions

import numpy as np
op_arr=np.arange(0,11);
#Array with array operations
print(op_arr+op_arr);
print(op_arr*op_arr);
print(op_arr-op_arr);
#Array with scalar operations
print(op_arr+100);
print(op_arr*100);
print(op_arr-100);
print(0/op_arr); # 0/0 results nan
print(1/op_arr); # 1/0 results inf which symbolises infinity
print(op_arr**2);
#Universal Array functions
print(np.sqrt(op_arr)); # Finds sqare root of all the elements in the array
print(np.exp(op_arr)); # Finds exponential function
print(np.max(op_arr)); # op_arr.max
print(np.sin(op_arr));
print(np.log(op_arr)); # log of 0 results -inf i.e -infinity
