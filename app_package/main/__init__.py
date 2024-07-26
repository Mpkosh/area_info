# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:03:31 2024

@author: user
"""

from flask import Blueprint


bp = Blueprint('main', __name__)


from app_package.main import routes