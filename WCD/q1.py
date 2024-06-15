# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 08:29:06 2024

@author: Ryan Tsai
"""

if __name__ == '__main__':
    num = [1,2,0,0]
    num = [5,6,7,3]
    num = [4,3,5,6]
    k = 34
    
    num = [1,8,2,0,4]
    k = 3859
    
    
    
    ndig = len(num)
    num_int = 0
    for d in range(ndig):
        num_int = num_int + num[d]*10**(ndig-1-d)
        
    num_out_int = num_int+k
    
    num_out = []
    for d in range(20):
        num_out_int = num_out_int/10
        val = round((num_out_int - int(num_out_int))*10)
        num_out = [val] + num_out
        num_out_int = int(num_out_int)
        if num_out_int == 0:
            break
        
    print(num_out)