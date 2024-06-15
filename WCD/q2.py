# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 08:52:57 2024

@author: Ryan Tsai
"""

if __name__ == '__main__':
    nums = [-1,0,1,2,-1,-4]
    
    out = []
    for i in range(len(nums)):
        for j in range(i+1,len(nums)):
            for k in range(j+1,len(nums)):
                if nums[i] + nums[j] + nums[k] == 0:
                    triple = [nums[i], nums[j], nums[k]]
                    triple.sort()
                    if not(out.count(triple)):
                        out.append(triple)
    
    print(out)