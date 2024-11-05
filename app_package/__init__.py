# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:02:42 2024

@author: user
"""

from flask import Flask, send_from_directory, send_file
#from flask_swagger import swagger
from config_f import Config
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
#from app_package.swagger import swaggerui_blueprint as bp_swagger
from flask_swagger_ui import get_swaggerui_blueprint



db = SQLAlchemy()
migrate = Migrate()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    #Swagger = swagger(app)    
    db.init_app(app)
    migrate.init_app(app, db)
    
    
    SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
    
    @app.route('/api/swagger.json')
    def swagger_json():
        # Read before use: http://flask.pocoo.org/docs/0.12/api/#flask.send_file
        return send_file('static/swagger.json')  
    
    swaggerui_blueprint = get_swaggerui_blueprint(
        SWAGGER_URL,
        '/api/swagger.json',
        config={"layout":"BaseLayout"}
    )
    app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)
    

    
    # blueprint registration
    from app_package.main import bp as bp_main
    app.register_blueprint(bp_main, url_prefix='/main')
    
    from app_package.api import bp as bp_api
    app.register_blueprint(bp_api, url_prefix='/api')
    '''
    
    SWAGGER_URL = '/docs'
    app.register_blueprint(bp_swagger, url_prefix=SWAGGER_URL)
    
    
    # swagger configs
    SWAGGER_URL = "/swagger"
    API_URL = "/static/swagger.json"
    SWAGGER_BLUEPRINT = get_swaggerui_blueprint(
        SWAGGER_URL,
        API_URL,
        config={
            "app_name": "api"
        }
    )
    app.register_blueprint(SWAGGER_BLUEPRINT, url_prefix=SWAGGER_URL)

    @app.route("/static/swagger.json")
    def specs():
        return send_from_directory(os.getcwd(), "static/swagger.json")
    '''

    return app


from app_package import models



