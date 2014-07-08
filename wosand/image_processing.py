import numpy as np
import math

def modify_matrix(matrix,beta=0.5,alpha=10):
    return sigmoid_contrast_enhancement(matrix,beta,alpha)
    
def linear_contrast_enhancement(matrix):
    mini = np.min(matrix)
    maxi = np.max(matrix)
    return (matrix-mini)/(maxi-mini)
    
def sigmoid_contrast_enhancement(matrix,beta=0.5,alpha=10,max=1):
    
    temp = (beta-matrix)/alpha
    temp_exp = np.exp(temp)
    final = 1/(1+temp_exp)*max
    return final
        
def sigmoid_local_contrast_enhancement(matrix,regions,alpha=1):
    copy = np.matrix(matrix)
    for reg in regions:
        # Extract region
        region = copy * reg
        # Apply the sigmoid filter
        beta = np.median(region[region>0])
        temp = (beta-matrix)/alpha
        temp_exp = np.exp(temp)
        # Modify the matrix
        matrix[reg>0] = 1/(1+temp[reg>0])
    return linear_contrast_enhancement(copy)

#def sharpen(matrix)

'''
def modify_matrix(matrix,regions):
    return linear_constrast(matrix,regions)
    
def linear_contrast_enhancement(matrix,regions):
    for (i,j) in regions:
        
        mini = np.min(matrix[matrix>0])
        maxi = np.max(matrix[matrix<1])
        return (matrix-mini)/(maxi-mini)
    
def sigmoid_contrast_enhancement(matrix):
    mini = np.min(matrix[matrix>0])
    maxi = np.max(matrix[matrix<1])
    return (matrix-mini)/(maxi-mini)
'''        