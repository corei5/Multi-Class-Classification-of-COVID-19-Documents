#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 15:17:54 2020

@author: tourist800
"""

def is_not_ascii(string):
    return string is not None and any([ord(s) >= 128 for s in string])
