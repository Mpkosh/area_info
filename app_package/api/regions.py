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
                            ValIdentityMatrix, MigForecast, \
                            DemForecast, ClusterInfo

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
    forecast_until = request.args.get('forecast_until', type = int, default=0)
    
    given_year = request.args.get('given_year', type = int, default=2020)
    territory_id = request.args.get('territory_id', type = int, default=34)
    n_age_groups = request.args.get('n_age_groups', type = int, default=5)
    unpack_after_70 = request.args.get('unpack_after_70', type = is_it_true, default=False)
    
    session = requests.Session()
    df = PopInfoForAPI.get_detailed_pop(session, territory_id, 
                                        unpack_after_70=unpack_after_70, last_year=False, 
                                        specific_year=0)
 
    last_pop_year = df.columns.levels[0][-1]

    # если нужен год больше текущего -- делаем прогноз
    if forecast_until>0:
        if forecast_until > last_pop_year:
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            forecast_df = DemForecast.MakeForecast(df, last_pop_year, 
                                                forecast_until - last_pop_year, folders)
        # отрезаем от прогноза первый год (== поданному на вход последнему году)
        df = pd.concat([df, forecast_df.iloc[:, 2:]], axis=1)
        # выбираем нужный интервал возрастов
        age_groups_df = PopulationInfo.age_groups(df, n_in_age_group=n_age_groups)

    else:
        if forecast_until > last_pop_year:
            folders={'popdir':file_dir,
                 'file_name':'Ленинградская область.xlsx'}
            df = DemForecast.MakeForecast(df, last_pop_year, 
                                                given_year - last_pop_year, folders)
        # выбираем нужный интервал возрастов
        age_groups_df = PopulationInfo.age_groups(df[given_year], n_in_age_group=n_age_groups)

    # чтобы мужчины были слева графика
    age_groups_df.iloc[:,::2] *= -1
    age_groups_df.index = age_groups_df.index.astype(str)
    
    return Response(age_groups_df.to_json(orient="split"), 
                    mimetype='application/json')


@bp_api.route('/regions/migration_data', methods=['GET'])
@cross_origin()
def migration_data():
    given_year = request.args.get('given_year', type = int, default=2020)
    territory_id = request.args.get('territory_id', type = int, default=34)
    n_age_groups = request.args.get('n_age_groups', type = int, default=5)
    
    
    session = requests.Session()
    df = PopInfoForAPI.get_detailed_pop(session, territory_id, 
                                                   unpack_after_70=True, last_year=False, 
                                                   specific_year=0)
    last_pop_year = df.columns.levels[0][-1]
    # если нужен год больше текущего -- делаем прогноз
    if given_year > last_pop_year:
        folders={'popdir':file_dir,
             'file_name':'Ленинградская область.xlsx'}
        df = DemForecast.MakeForecast(df, last_pop_year, 
                                            given_year - last_pop_year, folders)
    
    df_full = df[[given_year-1,given_year]]
    
 
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
    #print(df['geometry'])
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
    
    session = requests.Session()
    df = PopInfoForAPI.get_detailed_pop(session, territory_id, 
                                        unpack_after_70=True, last_year=False, 
                                        specific_year=0)
    last_pop_year = df.columns.levels[0][-1]
    # если нужен год больше текущего -- делаем прогноз
    if given_year > last_pop_year:
        folders={'popdir':file_dir,
             'file_name':'Ленинградская область.xlsx'}
        df = DemForecast.MakeForecast(df, last_pop_year, 
                                            given_year - last_pop_year, folders)
        
    # интервал возрастов == 1
    
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
    specific_year = request.args.get('year_agesex', type = int, default=2022)
    
    result = PopInfoForAPI.info(territory_id=territory_id, 
                                         show_level=show_level, specific_year=specific_year)
    
    return Response(result.set_geometry('geometry').to_json(), 
                    mimetype='application/json')
    
@bp_api.route('/regions/detailed_info', methods=['GET'])
@cross_origin()
def detailed_info():
    territory_id = request.args.get('territory_id', type = int, default=34)
    forecast_until = request.args.get('forecast_until', type = int, default=2024)
    
    pop_df, groups_df, dynamic_pop_df, \
        soc_pyramid_df, values_df = PopInfoForAPI.detailed_pop_info(territory_id, forecast_until)
    
    return [pop_df.to_json(), groups_df.to_json(), dynamic_pop_df.to_json(),
            soc_pyramid_df.to_json(), values_df.to_json()]


@bp_api.route('/regions/values_identities', methods=['GET'])
@cross_origin()
def values_identities():
    territory_id = request.args.get('territory_id', type = int, default = 34)
    feature_changed = request.args.get('feature_changed', type = is_it_true, default = False)
    changes_dict = request.args.get('changes_dict', type = str, default = "")
    result = ValIdentityMatrix.muni_tab(territory_id, feature_changed, changes_dict)
    return Response(result, mimetype='application/json')


@bp_api.route('/regions/children_values_identities', methods=['GET'])
@cross_origin()
def ch_values_identities():
    parent_id = request.args.get('parent_id', type = int, default = 1)
    show_level = request.args.get('show_level', type = int, default = 3)
    #level 3 - districts
    result = ValIdentityMatrix.ch_muni_tab(parent_id, show_level)
    return Response(result, mimetype='application/json')



#____________ OFFICIAL F21


@bp_api.route('/migrations/main_info', methods=['GET'])
@cross_origin()
def main_migr():
    territory_id = request.args.get('territory_id', type = int, default=34)
    show_level = request.args.get('show_level', type = int, default=2)
    with_mig_dest = request.args.get('mig_destinations', type = is_it_true, default=False)
    change_lo_level = request.args.get('change_lo_level', type = is_it_true, default=True)
    from_file = request.args.get('from_file', type = is_it_true, default=True)
    
    result = MigInfoForAPI.info(territory_id=territory_id, 
                                show_level=show_level, 
                                with_mig_dest=with_mig_dest,
                                change_lo_level=change_lo_level,
                                from_file=from_file)
    
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
    from_file = request.args.get('from_file', type = is_it_true, default=True)

    result = MigInfoForAPI.info(territory_id=territory_id, 
                                detailed=True, 
                                with_mig_dest=with_mig_dest,
                                md_year=md_year,
                                from_file=from_file)
                            
    if with_mig_dest:
        fin_df, from_to_geom, from_to_lines = result
        from_to_geom = from_to_geom.set_geometry('geometry')
        from_to_lines = from_to_lines.set_geometry('line')
        return [fin_df.to_json(orient="records"), from_to_geom.to_json(), from_to_lines.to_json()]
    else:
        return Response(result.to_json(orient="records"), 
                        mimetype='application/json')


@bp_api.route('/migrations/forecast', methods=['GET'])
@cross_origin()
def mig_forecast():
    features = ['year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 
                'foodseats', 'retailturnover', 'livarea', 'sportsvenue', 
                'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 
                'hospitals', 'beforeschool']
    
    input_values = []
    for param in features:
        param_value = request.args.get(param, type = float)
        input_values.append(param_value)
    
    # обработка входных параметров
    inputdata = pd.DataFrame.from_records([input_values], 
                                          columns=features)
    inputdata = inputdata.astype(float)
    
    res = MigForecast.model_outcome(inputdata)
    res_df = pd.DataFrame([res], columns=['popsize', 'saldo'])
    return Response(res_df.to_json(orient="records"), 
                    mimetype='application/json')


# ___________________ Cluster analysis

# определить кластер для входных данных
@bp_api.route('/cluster_info/cluster', methods=['GET'])
@cross_origin()
def cluster():
    features = ['type','profile','year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 
                'foodseats', 'retailturnover', 'livarea', 'sportsvenue', 
                'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 
                'hospitals', 'beforeschool']
    
    input_values = []
    for param in features:
        param_value = request.args.get(param)
        input_values.append(param_value)
    
    # обработка входных параметров
    inputdata = pd.DataFrame.from_records([input_values], 
                                          columns=features)
    inputdata.iloc[:,2:] = inputdata.iloc[:,2:].astype(float)
    
    
    res = ClusterInfo.whatcluster(inputdata)
    return res
    #return Response(res_df.to_json(orient="records"),  mimetype='application/json')


# поиск наиболее близки поселений на основе социально-экономических индикаторов
@bp_api.route('/cluster_info/top10_closest', methods=['GET'])
@cross_origin()
def top10_closest():
    features = ['type','profile','year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 
                'foodseats', 'retailturnover', 'livarea', 'sportsvenue', 
                'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 
                'hospitals', 'beforeschool']
    
    input_values = []
    for param in features:
        param_value = request.args.get(param)
        input_values.append(param_value)
    
    # обработка входных параметров
    inputdata = pd.DataFrame.from_records([input_values], 
                                          columns=features)
    inputdata.iloc[:,2:] = inputdata.iloc[:,2:].astype(float)
    
    
    res = ClusterInfo.siblingsfinder(inputdata)
    return Response(res.to_json(orient="records"),  
                    mimetype='application/json')


# разница от наиболее близкого из лучшего кластера
@bp_api.route('/cluster_info/diff_from_closest', methods=['GET'])
@cross_origin()
def diff_from_closest():
    features = ['type','profile','year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 
                'foodseats', 'retailturnover', 'livarea', 'sportsvenue', 
                'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 
                'hospitals', 'beforeschool']
    
    input_values = []
    for param in features:
        param_value = request.args.get(param)
        input_values.append(param_value)
    
    # обработка входных параметров
    inputdata = pd.DataFrame.from_records([input_values], 
                                          columns=features)
    inputdata.iloc[:,2:] = inputdata.iloc[:,2:].astype(float)
    
    
    res = ClusterInfo.headtohead(inputdata)
    return Response(res.to_json(),  
                    mimetype='application/json')


# разница от наиболее близкого из лучшего кластера
@bp_api.route('/cluster_info/change_plan', methods=['GET'])
@cross_origin()
def change_plan():
    features = ['type','profile','year', 'popsize', 'avgemployers', 'avgsalary', 'shoparea', 
                'foodseats', 'retailturnover', 'livarea', 'sportsvenue', 
                'servicesnum', 'roadslen', 'livestock', 'harvest', 'agrprod', 
                'hospitals', 'beforeschool']
    
    input_values = []
    for param in features:
        param_value = request.args.get(param)
        input_values.append(param_value)
    
    # обработка входных параметров
    inputdata = pd.DataFrame.from_records([input_values], 
                                          columns=features)
    inputdata.iloc[:,2:] = inputdata.iloc[:,2:].astype(float)
    
    
    res = ClusterInfo.reveal(inputdata)

    return Response(res.to_json(orient="records"),  
                    mimetype='application/json')