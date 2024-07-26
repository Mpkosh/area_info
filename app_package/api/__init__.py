# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:09:38 2024

@author: user
"""
from flask import Blueprint


bp = Blueprint('api', __name__)


from app_package.api import regions
