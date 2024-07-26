# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:02:42 2024

@author: user
"""

from flask import Flask
from config_f import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


db = SQLAlchemy()
migrate = Migrate()
    

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    
    db.init_app(app)
    migrate.init_app(app, db)
    # blueprint registration
    from app_package.main import bp as bp_main
    app.register_blueprint(bp_main, url_prefix='/main')
    
    from app_package.api import bp as bp_api
    app.register_blueprint(bp_api, url_prefix='/api')
    
    return app


from app_package import models



