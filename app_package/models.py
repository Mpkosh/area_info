# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:04:27 2024

@author: user
"""

#from typing import Optional # column that is allowed to be empty or nullable
import sqlalchemy as sa
import sqlalchemy.orm as so
from app_package import db


class Region(db.Model):
    id: so.Mapped[int] = so.mapped_column(primary_key=True)
    okato_id: so.Mapped[str] = so.mapped_column(sa.String(11))
    #pop_data: so.Mapped[str] = so.mapped_column(sa.String(80)) # имя файла
    
    def __repr__(self):
       return f'Region {self.okato_id}'