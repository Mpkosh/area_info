# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:52:24 2024

@author: user
"""
from app_package import create_app, db
from app_package.models import Region
import sqlalchemy as sa
import sqlalchemy.orm as so


app = create_app()

# for flask shell
@app.shell_context_processor
def make_shell_context():
    return {'sa':sa, 'so':so, 'db':db, 'Region':Region}
    

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0')