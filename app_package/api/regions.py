# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:11:49 2024

@author: user
"""
from flask import request, send_file, redirect, Response
from flask_cors import CORS, cross_origin

from app_package.api import bp as bp_api
from app_package.src import PreproDF, PopulationInfo, \
                            AreaOnMapFile, DensityInfo, \
                            PopInfoForAPI, MigInfoForAPI, \
                            ValIdentityMatrix

import pandas as pd
import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import Polygon
import os



file_dir = 'app_package/src/population_data/'
social_api = os.environ.get('SOCIAL_API')
territories_api = os.environ.get('TERRITORY_API') 


@bp_api.route('/')
@cross_origin()
def zero_to_docs():
    
    return redirect("/api/docs", code=302)


@bp_api.route('/here')
@cross_origin()
def index():
    print('1',social_api)
    print('2', os.environ.get('SOCIAL_API'))
    return 'hey from api'


def func(x):
    x1, x2 = x[0], x[1]
    
    if x1 == 100:
        return 100
    elif x1 != x2:
        return f'{int(x1)}-{int(x2)}'
    else:
        return int(x1)


def is_it_true(value):
  return value.lower() == 'true'


@bp_api.route('/regions/pyramid_data', methods=['GET'])
@cross_origin()
def pyramid_data():
    given_year = request.args.get('given_year', type = int, default=2020)
    territory_id = request.args.get('territory_id', type = int, default=34)
    n_age_groups = request.args.get('n_age_groups', type = int, default=5)
    
    try:
        url = social_api + f'indicators/2/{territory_id}/detailed'
        r = requests.get(url, params={'year':given_year})
        all_popul = r.json()
        df = pd.DataFrame(all_popul[0]['data'])
        # на всякий случай задаем возраста: 
            # 1) все с интервалом 1 год
            # 2) 100+ 
            # 3) с интервалом 4 года для старших
        df = df[(df['age_start']==df['age_end'])|(
                df['age_start']==100)|(
                (df['age_end']-df['age_start']==4)&(df['age_end'].isin([74,79,84,
                                                                        89,94,99])))]
        df['группа'] = df.iloc[:,:2].apply(func, 1)
        df = df.set_index('группа').iloc[:,2:]

    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    # ставим года
    df.columns = pd.MultiIndex.from_product([[given_year], ['Мужчины', 'Женщины']])
    df.columns.set_names(['', "пол"], level=[0,1], inplace=True)
    df.bfill(inplace=True)
    
    df = PreproDF.add_ages_70_to_100(df)
    df.index = df.index.astype(str)
    # уберем возрастные интервалы
    df = df[df.index.isin([str(i) for i in range(0,101)])]
    
    df.index = df.index.astype(int)
    df.sort_index(inplace=True)
    
    # выбираем нужный интервал возрастов
    age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)
    
    # чтобы мужчины были слева графика
    age_groups_df[given_year]['Мужчины'] *= -1
    age_groups_df.index = age_groups_df.index.astype(str)
    
    return Response(age_groups_df[given_year].to_json(orient="split"), 
                    mimetype='application/json')


@bp_api.route('/regions/migration_data', methods=['GET'])
@cross_origin()
def migration_data():
    given_year = request.args.get('given_year', type = int, default=2020)
    territory_id = request.args.get('territory_id', type = int, default=34)
    n_age_groups = request.args.get('n_age_groups', type = int, default=5)
    
    try:
        url = social_api + f'indicators/2/{territory_id}/detailed'
        
        dfs = []
        # нужны данные за два года
        for year in [given_year-1, given_year]:
            r = requests.get(url, params={'year':year})
            all_popul = r.json()
            
            df = pd.DataFrame(all_popul[0]['data'])
            # на всякий случай задаем возраста: 
                # 1) все с интервалом 1 год
                # 2) 100+ 
                # 3) с интервалом 4 года для старших
            df = df[(df['age_start']==df['age_end'])|(
                    df['age_start']==100)|(
                    (df['age_end']-df['age_start']==4)&(df['age_end'].isin([74,79,84,
                                                                            89,94,99])))]
            df['группа'] = df.iloc[:,:2].apply(func, 1)
            df = df.set_index('группа').iloc[:,2:]
            df.columns = ['Мужчины', 'Женщины']
            dfs.append(df)
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    df_full = pd.concat(dfs, axis=1, ignore_index=True) 
    # ставим года
    df_full.columns = pd.MultiIndex.from_tuples([(given_year-1, "Мужчины"),(given_year-1, 'Женщины'),
                                                       (given_year, "Мужчины"),(given_year, 'Женщины')])
    df_full.columns.set_names(['', "пол"], level=[0,1], inplace=True)
    df_full.bfill(inplace=True)
    
    df_full = PreproDF.add_ages_70_to_100(df_full)
    df_full.index = df_full.index.astype(str)
    # уберем возрастные интервалы
    df_full = df_full[df_full.index.isin([str(i) for i in range(0,101)])]
    
    df_full.index = df_full.index.astype(int)
    df_full.sort_index(inplace=True)
    
    # вычисляем сальдо
    difference_df = PopulationInfo.expected_vs_real(df_full,
                                                    morrate_file=file_dir+'morrate.xlsx')
    # группируем значения по возрастам
    diff_grouped = PopulationInfo.group_by_age(difference_df, 
                                               n_in_age_group=n_age_groups)
    # выбираем нужный год
    required_part = diff_grouped.loc[:,given_year]
    
    return Response(required_part.to_json(orient="split"), 
                    mimetype='application/json')


def create_polygon(coordinates):
    return Polygon(coordinates[0])


@bp_api.route('/regions/density_data', methods=['GET'])
@cross_origin()
def density_data():
    parent_id = request.args.get('parent_id', type = str, default=34)
    given_year = request.args.get('given_year', type = int, default=2020)
    last_only = request.args.get('last_only', type = is_it_true, default=True)
    
    try:
        # берем координаты территорий внутри заданной 
        url = territories_api + 'api/v1/territories'
        params = {'parent_id':parent_id,'get_all_levels':'false','page':'1','page_size':'1000'}
        r = requests.get(url, params = params)
        places_df = pd.DataFrame(r.json()['results'])
        
        if last_only:
            places_popul = pd.json_normalize(places_df['properties'])['Численность населения'].values
            given_year=2023 # не прописано в ответе, но данные за посл.год
            places_df[given_year] = places_popul
        else:
            # ищем данные о суммарной популяции каждой тер-рии за заданный год
            # (тк в данных с координанатами население только за последний год?)
            places_ids = places_df['territory_id'].values
            places_popul = []
            for place_id in places_ids:
                url = social_api + f'indicators/2/{place_id}'
                r = requests.get(url)
                all_popul = r.json()
                # выбираем данные за нужный год
                needed_popul = [i['value'] for i in all_popul if i['year']==given_year]
                places_popul.append(needed_popul[0])
            places_df[given_year] = places_popul
            
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    # создаем геодатафрейм и вставляем данные по координатам
    df = gpd.GeoDataFrame(places_df)
    print(df['geometry'])
    df['geometry'] = df['geometry'].apply(lambda x: create_polygon(x['coordinates'][0]))
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
    res = df[['territory_id','name','geometry',given_year, clm_with_dnst,'binned']]
    
    return Response(res.to_json(), 
                    mimetype='application/json') 


@bp_api.route('/regions/density_data_full', methods=['GET'])
@cross_origin()
def density_data_full():
    parent_id = request.args.get('parent_id', type = str, default=34)

    session = requests.Session()
    
    full_df = DensityInfo.density_data_geojson(session=session, territory_id=parent_id, 
                                                 from_api=True)
    return Response(full_df.to_json(), 
                    mimetype='application/json') 


@bp_api.route('/regions/area_needs', methods=['GET'])
@cross_origin()
def area_needs():
    territory_id = request.args.get('territory_id', type = int, default=34)
    given_year = request.args.get('given_year', type = int, default=2020)
    
    try:
        url = social_api + f'indicators/2/{territory_id}/detailed'
        r = requests.get(url, params={'year':given_year})
        all_popul = r.json()
        df = pd.DataFrame(all_popul[0]['data'])
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
        
    df['группа'] = df.iloc[:,:2].apply(func, 1)
    df = df.set_index('группа').iloc[:,2:]

    # ставим года
    df.columns = pd.MultiIndex.from_product([[given_year], ['Мужчины', 'Женщины']])
    df.columns.set_names(['', "пол"], level=[0,1], inplace=True)
    df.bfill(inplace=True)
    
    df = PreproDF.add_ages_70_to_100(df)
    df.index = df.index.astype(str)
    # уберем возрастные интервалы
    df = df[df.index.isin([str(i) for i in range(0,101)])]
    
    df.index = df.index.astype(int)
    df.sort_index(inplace=True)

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
    return Response(fc.apply(lambda x: x/fc.sum(1)).to_json(orient="split"), 
                    mimetype='application/json')


@bp_api.route('/regions/pop_needs', methods=['GET'])
@cross_origin()
def pop_needs():
    df = pd.read_csv(file_dir+'pop_needs.csv', index_col=0)
    return Response(df.to_json(orient="split"), 
                    mimetype='application/json')

#____________ OFFICIAL F11

@bp_api.route('/regions/main_info', methods=['GET'])
@cross_origin()
def main_info():
    territory_id = request.args.get('territory_id', type = int, default=34)
    show_level = request.args.get('show_level', type = int, default=2)
    
    result = PopInfoForAPI.info(territory_id=territory_id, 
                                         show_level=show_level)
    
    return Response(result.set_geometry('geometry').to_json(), 
                    mimetype='application/json')
    
@bp_api.route('/regions/detailed_info', methods=['GET'])
@cross_origin()
def detailed_info():
    territory_id = request.args.get('territory_id', type = int, default=34)
    pop_df, groups_df, dynamic_pop_df, \
        soc_pyramid_df, values_df = PopInfoForAPI.detailed_pop_info(territory_id)
    
    return [pop_df.to_json(), groups_df.to_json(), dynamic_pop_df.to_json(),
            soc_pyramid_df.to_json(), values_df.to_json()]

#____________ OFFICIAL F21


@bp_api.route('/migrations/main_info', methods=['GET'])
@cross_origin()
def main_migr():
    territory_id = request.args.get('territory_id', type = int, default=34)
    show_level = request.args.get('show_level', type = int, default=2)
    with_mig_dest = request.args.get('mig_destinations', type = is_it_true, default=False)
    change_lo_level = request.args.get('change_lo_level', type = is_it_true, default=True)
    result = MigInfoForAPI.info(territory_id=territory_id, 
                                show_level=show_level, 
                                with_mig_dest=with_mig_dest,
                                change_lo_level=change_lo_level)
    
    if with_mig_dest:
        fin_df, from_to_geom, from_to_lines = result
        fin_df = gpd.GeoDataFrame(fin_df).set_geometry('geometry')
        from_to_geom = from_to_geom.set_geometry('geometry')
        from_to_lines = from_to_lines.set_geometry('line')
        return [fin_df.to_json(), from_to_geom.to_json(), from_to_lines.to_json()]
    else:
        return Response(result.set_geometry('geometry').to_json(), 
                        mimetype='application/json')


@bp_api.route('/migrations/detailed_info', methods=['GET'])
@cross_origin()
def detailed_migr():
    territory_id = request.args.get('territory_id', type = int, default=34)
    with_mig_dest = request.args.get('mig_destinations', type = is_it_true, default=False)
    md_year = request.args.get('given_year', type = int, default=2022)
    result = MigInfoForAPI.info(territory_id=territory_id, 
                                detailed=True, 
                                with_mig_dest=with_mig_dest,
                                md_year=md_year)
                            
    if with_mig_dest:
        fin_df, from_to_geom, from_to_lines = result
        from_to_geom = from_to_geom.set_geometry('geometry')
        from_to_lines = from_to_lines.set_geometry('line')
        return [fin_df.to_json(orient="records"), from_to_geom.to_json(), from_to_lines.to_json()]
    else:
        return Response(result.to_json(orient="records"), 
                        mimetype='application/json')


@bp_api.route('/regions/values_identities', methods=['GET'])
@cross_origin()
def values_identities():
    territory_id = request.args.get('territory_id', type = int, default = 34)
    result = ValIdentityMatrix.muni_tab(territory_id)
    return Response(result, mimetype='application/json')
