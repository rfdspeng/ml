# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 08:04:43 2024

@author: Ryan Tsai
"""

import re

if __name__ == '__main__':
    greeting = 'Hello! We are re super excited to have you join our AI bootcamp to learn about NLP amongst other things!'
    
    greeting = greeting.lower()
    outstr = re.split(' |! *|we|are|re|to|have|you|our|ai|about|other',greeting)
    
    outdict = {}
    for s in outstr:
        if s == '':
            continue
        if not(s in outdict):
            outdict[s] = 0
        outdict[s] += 1
    
    print(outdict)