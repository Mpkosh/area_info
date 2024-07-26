# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:11:49 2024

@author: user
"""
from app_package.api import bp as bp_api
from flask import request, send_file
import pandas as pd
import geopandas as gpd

import numpy as np
import requests
from shapely.geometry import Polygon
from app_package import db
import sqlalchemy as sa
from app_package.models import Region
#from app.api.errors import bad_request
from app_package.src import PreproDF, PopulationInfo, AreaOnMapFile
from app_package.translate import translate
from flask_cors import CORS, cross_origin


file_dir = 'app_package/src/population_data/'
geo_dir = 'app_package/src/geo_data/'
csv_path = 'app_package/src/data_each_reg/'
okato_name_dict = dict({'41206000000':'Волосовский',
                        '41209000000':'Волховский',
                        '41212000000':'Всеволожский'})


def region_from_db(okato_id):
    # находим нужный регион
    data = db.first_or_404(sa.select(Region).where(Region.okato_id == okato_id))
    # предобрабатываем данные о населении
    orig_df = PreproDF.df_from_db_path(csv_path + data.okato_id+'.csv')
    df = PreproDF.prepro_df(orig_df, morrate_file=file_dir+'morrate.xlsx',
                            aux_clm_file = file_dir+'correct_clm_sex.csv')
    return df


@bp_api.route('/here')
def index():
    return 'hey from api'

@bp_api.route('/here_get')
def q():
    okato_id = request.args.get('okato_id', type = str)
    data = db.first_or_404(sa.select(Region).where(Region.okato_id == okato_id))
    return f'given id is {okato_id}, data is {data}'

@bp_api.route('/regions/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    return {'text': translate(data['text'])}


@bp_api.route('/regions/pyramid_plot', methods=['GET'])
def pyramid_plot():
    given_year = request.args.get('given_year', type = int)
    territory_id = request.args.get('territory_id', type = int)
    #n_age_groups = request.args.get('n_age_groups', default=5, type = int)
    
    
    url = f'https://socdemo.lok-tech.com/indicators/2/{territory_id}/detailed'
    r = requests.get(url, params={'year':given_year})
    all_popul = r.json()
    df = pd.DataFrame(all_popul[0]['data'])
    df['группа'] = df.iloc[:,:2].apply(func, 1)
    df = df.set_index('группа').iloc[:,2:]
    df.columns = ['Мужчины', 'Женщины']
    # выбираем нужный интервал возрастов
    n_age_groups = 5
    age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)
    
    # ставим год, чтобы не менять код в функциях
    age_groups_df.columns = pd.MultiIndex.from_tuples([(given_year, "Мужчины"),(given_year, 'Женщины')])
    # рисуем график
    bytes_obj  = PopulationInfo.plot_population_info(age_groups_df, 
                                                     chosen_years=[given_year], 
                                                     area_name='', 
                                                     figsize=(10,13))
    
    return send_file(bytes_obj, download_name='pyr_plot.png',mimetype='image/png')


def func(x):
    x1, x2 = x[0], x[1]
    
    if x1 != x2:
        return f'{x1}-{x2}'
    else:
        return x1
    

@bp_api.route('/regions/pyramid_data', methods=['GET'])
def pyramid_data():
    given_year = request.args.get('given_year', type = int)
    territory_id = request.args.get('territory_id', type = int)
    #n_age_groups = request.args.get('n_age_groups', default=5, type = int)
    
    
    url = f'https://socdemo.lok-tech.com/indicators/2/{territory_id}/detailed'
    r = requests.get(url, params={'year':given_year})
    all_popul = r.json()
    df = pd.DataFrame(all_popul[0]['data'])
    df['группа'] = df.iloc[:,:2].apply(func, 1)
    df = df.set_index('группа').iloc[:,2:]
    df.columns = ['Мужчины', 'Женщины']
    
    '''
    df = PreproDF.prepro_df(qq, from_api=True, morrate_file=file_dir+'morrate.xlsx',
                            aux_clm_file = file_dir+'correct_clm_sex.csv')
    '''
    # выбираем нужный интервал возрастов
    n_age_groups = 5
    age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)
    
    # чтобы мужчины были слева графика
    age_groups_df['Мужчины'] *= -1
    
    return age_groups_df.to_json(orient="split")


@bp_api.route('/regions/migration_plot', methods=['GET'])
def migration_plot():
    okato_id = request.args.get('okato_id', type = str)
    given_year = request.args.get('given_year', type = int)
    n_age_groups = request.args.get('n_age_groups', default=5, type = int)
    df = region_from_db(okato_id)

    # вычисляем сальдо
    difference_df = PopulationInfo.expected_vs_real(df, morrate_file=file_dir+'morrate.xlsx')
    # группируем значения по возрастам
    diff_grouped = PopulationInfo.group_by_age(difference_df, 
                                               n_in_age_group=n_age_groups)
    # рисуем разницы
    bytes_obj = PopulationInfo.plot_difference_info(diff_grouped, 
                                                    area_name = okato_name_dict[okato_id], 
                                                    chosen_year=given_year,
                                                    figsize=(16,7))
    
    return send_file(bytes_obj, download_name='mig_plot.png',mimetype='image/png')


@bp_api.route('/regions/migration_data', methods=['GET'])
def migration_data():
    given_year = request.args.get('given_year', type = int)
    territory_id = request.args.get('territory_id', type = int)
    
    url = f'https://socdemo.lok-tech.com/indicators/2/{territory_id}/detailed'
    
    # нужны данные за два года
    for year in [given_year-1, given_year]:
        r = requests.get(url, params={'year':year})
        all_popul = r.json()
        
        df = pd.DataFrame(all_popul[0]['data'])
        df['группа'] = df.iloc[:,:2].apply(func, 1)
        df = df.set_index('группа').iloc[:,2:]
        df.columns = ['Мужчины', 'Женщины']
        # выбираем нужный интервал возрастов
        n_age_groups = 5
        age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)
        # ставим год, чтобы не менять код в функциях
        age_groups_df.columns = pd.MultiIndex.from_tuples([(given_year, "Мужчины"),(given_year, 'Женщины')])
    
    
    # вычисляем сальдо
    difference_df = PopulationInfo.expected_vs_real(df,
                                                    morrate_file=file_dir+'morrate.xlsx')
    # группируем значения по возрастам
    diff_grouped = PopulationInfo.group_by_age(difference_df, 
                                               n_in_age_group=n_age_groups)
    # выбираем нужный год
    required_part = diff_grouped.loc[:,given_year]
    
    return required_part.to_json(orient="split")


def create_polygon(coordinates):
    return Polygon(coordinates[0])

# todo: return as a layer
@bp_api.route('/regions/density_map', methods=['GET'])
def density_map():
    okato_id = request.args.get('okato_id', type = str)
    given_year = request.args.get('given_year', type = int)
    area_name = okato_name_dict[okato_id]
    #df = region_from_db(okato_id)
    
    # получаем данные из файла
    d = AreaOnMapFile.get_area_from_file(geojson_filename='', area_name=okato_id,  
                                         choose_from_files=True, 
                                         area_files_path=geo_dir, 
                                         temp_no_water_file=geo_dir+\
                                         'Границы_только_МР_Границы_ЛО_Без_воды.geojson')
    # считаем плотность населения
    d_with_dnst = AreaOnMapFile.calculate_density(d, area_name=area_name, clm_area_name='name')
    # рисуем плотность населения на карте
    bytes_obj = AreaOnMapFile.plot_density(d_with_dnst, year=given_year, clm_with_name='name', 
                                           area_name=area_name)
    
    return send_file(bytes_obj, download_name='map_plot.png',mimetype='image/png')


@bp_api.route('/regions/area_needs', methods=['GET'])
@cross_origin()
def area_needs():
    territory_id = request.args.get('territory_id', type = int)
    given_year = request.args.get('given_year', type = int)
    
    url = f'https://socdemo.lok-tech.com/indicators/2/{territory_id}/detailed'
    r = requests.get(url, params={'year':given_year})
    all_popul = r.json()
    df = pd.DataFrame(all_popul[0]['data'])
    df['группа'] = df.iloc[:,:2].apply(func, 1)
    df = df.set_index('группа').iloc[:,2:]
    df.columns = ['Мужчины', 'Женщины']
    # выбираем нужный интервал возрастов
    n_age_groups = 5
    #age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)
    
    #_______________________________
    
    #okato_id = request.args.get('okato_id', type = str)
    #given_year = request.args.get('given_year', type = int)
    #df = region_from_db(okato_id)

    # потребности
    df_needs = pd.read_csv(file_dir+'pop_needs.csv', index_col=0)
    needs = df_needs.columns[1:]
    age_groups = df_needs['Возраст']
    values = df_needs.iloc[:,1:].values

    # высчитываем от каждой возрастной группы
    all_values = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
    k_temp = df[given_year]
    all_count = k_temp.sum().sum()
    for age_group, value in zip(age_groups[::-1], values[::-1]):
        start = int(age_group.split('-')[0])
        finish = int(age_group.split('-')[1])
        percent = k_temp.loc[start:finish].abs().sum().sum() / all_count
        all_values += np.array(value) * percent

    fc = pd.DataFrame([all_values], columns=needs)
    return fc.apply(lambda x: x/fc.sum(1)).to_json(orient="split")


@bp_api.route('/regions/pop_needs', methods=['GET'])
@cross_origin()
def pop_needs():
    df = pd.read_csv(file_dir+'pop_needs.csv', index_col=0)
    return df.to_json(orient="split")


@bp_api.route('/regions/density_data', methods=['GET'])
def density_data():
    parent_id = request.args.get('parent_id', type = str)
    given_year = request.args.get('given_year', type = int)
    
    # берем координаты территорий внутри заданной 
    url = 'https://urban-api-107.idu.kanootoko.org/api/v1/territories'
    params = {'parent_id':parent_id,'get_all_levels':'false','page':'1','page_size':'50'}
    r = requests.get(url, params = params)
    places_df = pd.DataFrame(r.json()['results'])
    
    # ищем данные о суммарной популяции каждой тер-рии за заданный год
    # (тк в данных с координанатами население только за последний год?)
    places_ids = places_df['territory_id'].values
    places_popul = []
    for place_id in places_ids:
        url = f'https://socdemo.lok-tech.com/indicators/2/{place_id}'
        r = requests.get(url)
        all_popul = r.json()
        # выбираем данные за нужный год
        needed_popul = [i['value'] for i in all_popul if i['year']==given_year]
        places_popul.append(needed_popul[0])
    
    places_df[given_year] = places_popul
    
    # создаем геодатафрейм и вставляем данные по координатам
    df = gpd.GeoDataFrame(places_df)
    df['geometry'] = df['geometry'].apply(lambda x: create_polygon(x['coordinates']))
    df['geometry'].crs = 'EPSG:4326'
    df = df.set_geometry('geometry')
    # считаем плотность населения
    df = AreaOnMapFile.calculate_density(df)
    # задаем колонку
    clm_with_dnst = f'{given_year}_dnst'
    
    labels_ordered = ['0 — 10','10 — 100','100 — 500',
                  '500 — 1 000','1 000 — 5 000','5 000 — 10 000']
    # долго выполняются след.строки
    df['binned'] = pd.cut(df[clm_with_dnst], 
                        bins=[0,10,100,500,1000,5000,10000],
                        labels=labels_ordered)
    df = df.sort_values('binned')
    # тк иначе жалуется, если отсутствует категория
    df['binned'] = df['binned'].astype('str') 
    
    #return df[['fid','name',str(given_year), 'geometry', clm_with_dnst,'binned']].to_json()
    return df[['territory_id','name','geometry',given_year, clm_with_dnst,'binned']].to_json()