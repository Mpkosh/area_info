# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:04:18 2024

@author: user
"""
from flask import render_template, request
from app_package.main import bp as bp_main

@bp_main.route('/')
@bp_main.route('/index')
def index():
    return 'hey'

# по умолчанию только GET 
@bp_main.route('/region_data')
def region_data():
    okato_id = request.args.get('okato_id', type = str)
    area_name = request.args.get('area_name', type = str)
    return render_template('index.html', the_word='recieved!', okato_id=okato_id, 
                           area_name=area_name)
