# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:00:12 2024

@author: user
"""

import requests


if __name__ == '__main__':
    url = 'https://socdemo.lok-tech.com/indicators/'
    url = 'https://urban-api-107.idu.kanootoko.org/api/v1/territories'
    
    params = {'skip':'0', 'limit':'10'}
    params = {'parent_id':'1', 'get_all_levels':'false','page':'1','page_size':'20'}
    
    r = requests.get(url, params = params)
    
    print(r.json()['results'])