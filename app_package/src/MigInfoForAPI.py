# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:21:24 2024

@author: user
"""

import requests
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning, 
                        message=r".*Passing a BlockManager.*")
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import os
#import networkx as nx
from itertools import permutations


social_api = os.environ.get('SOCIAL_API')
terr_api = os.environ.get('TERRITORY_API') 
file_path = 'app_package/src/for_mig_api/'


class Territory():
    
    def __init__(self, territory_id=34):
        self.territory_id = territory_id
        self.df = pd.DataFrame([])
        self.df['territory_id'] = [self.territory_id]
        self.children = []
        self.parent = 0
        

def create_point(x):
    return Point(x['coordinates'])       


def info_for_graph(territory_id, show_level=0, md_year=2022):
    session = requests.Session()
    current_territory = Territory(territory_id)
    main_info(session, current_territory, show_level)
    
    if show_level < current_territory.territory_type:
        raise ValueError(f'Show level must be >= territory type; given show_level={show_level} and territory_type={current_territory.territory_type}')

    n_children = show_level - current_territory.territory_type
    terr_classes = [current_territory]
    terr_ids = current_territory.df['territory_id'].values
    # если нужен уровень детальнее
    for i in range(n_children):
        for ter_id, ter_class in zip(terr_ids, terr_classes):
            get_children(session, ter_id, ter_class)
        # новый набор для итерации    
        terr_classes = [child for one_class in terr_classes for child in one_class.children]
        # новые id тер-рий
        terr_ids = [one_class.territory_id for one_class in terr_classes]

    # all final children in <terr_classes>    
    fin_df = pd.DataFrame([])
    fin_df['territory_id'] = [cl.territory_id for cl in terr_classes]
    fin_df['oktmo'] = [cl.oktmo for cl in terr_classes]
    fin_df['name'] = [cl.name for cl in terr_classes]
    fin_df['geometry'] = [cl.geometry for cl in terr_classes]
    fin_df['parent_id'] = [cl.parent.territory_id for cl in terr_classes]
    
    return fin_df


def info(territory_id, show_level=0, detailed=False, 
         with_mig_dest=False, fill_in_4=False, md_year=2022, 
         change_lo_level=True, from_file=True):
    
    session = requests.Session()
    current_territory = Territory(territory_id)
    main_info(session, current_territory, show_level)
    
    if with_mig_dest:
        
        if (show_level == 4):
            raise NotImplementedError('Migration destinations for show_level=4 are not available yet')
    
        if (show_level == 1) & (md_year not in [2019,2020,2021,2022,2023]):
             raise NotImplementedError('Migration destinations for show_level=1 are available for years 2019-2023; given {md_year}')
             
        if md_year not in [2019,2020,2021,2022]:
            raise ValueError(f'Migration destinations for show_level>1 are available for years 2019-2022, given {md_year}')
    
            
    if detailed:
        fin = pd.DataFrame([])
        fin['territory_id'] = [current_territory.territory_id]
        fin['oktmo'] = [current_territory.oktmo]
        fin['name'] = [current_territory.name]
        fin['geometry'] = [current_territory.geometry]
        fin['centre_point'] = [current_territory.centre_point]
        
        # добавляем направления миграции
        if with_mig_dest:
            # для detailed_info ставится show_level=0
            # для уровня областей быстрее из файла (surpise-surprise)
            if from_file & (show_level <= 1):
                
                from_to_geom, from_to_lines = mig_dest_prepared(show_level=show_level, 
                                                                fin_df=fin, siblings=[], 
                                                                change_lo_level=change_lo_level, 
                                                                md_year=md_year,
                                                                from_file=from_file)
            else:    
                # собираем территории того же родителя
                n_children = 1
                current_territory.parent.territory_type = current_territory.territory_type-1
                terr_classes = [current_territory.parent]
                terr_ids = [current_territory.parent.territory_id]
                # если нужен уровень детальнее
                for i in range(n_children):
                    for ter_id, ter_class in zip(terr_ids, terr_classes):
                        get_children(session, ter_id, ter_class)
                    # новый набор для итерации    
                    terr_classes = [child for one_class in terr_classes for child in one_class.children]
                    # новые id тер-рий
                    terr_ids = [one_class.territory_id for one_class in terr_classes]
    
                siblings = pd.DataFrame([])
                siblings['territory_id'] = [cl.territory_id for cl in terr_classes]
                siblings['oktmo'] = [cl.oktmo for cl in terr_classes]
                siblings['name'] = [cl.name for cl in terr_classes]
                siblings['geometry'] = [cl.geometry for cl in terr_classes]
                siblings['centroid'] = [cl.centre_point for cl in terr_classes]
    
                show_level = terr_classes[0].territory_type
                from_to_geom, from_to_lines = mig_dest_prepared(show_level=show_level, 
                                                                fin_df=fin, siblings=siblings, 
                                                                change_lo_level=change_lo_level,
                                                               md_year=md_year)

        fin_df = pd.DataFrame([])
        
        fin_df = detailed_migration(session, current_territory, fin_df)
        fin_df = fin_df.rename_axis('year').reset_index()
        fin_df = detailed_factors(session, current_territory, fin_df)
        #fin_df['Численность населения'] = [current_territory.pop_all]
      
        
    # for main_info    
    else:
        if show_level < current_territory.territory_type:
            raise ValueError(f'Show level must be >= territory type; given show_level={show_level} and territory_type={current_territory.territory_type}')

        n_children = show_level - current_territory.territory_type
        terr_classes = [current_territory]
        terr_ids = current_territory.df['territory_id'].values
        # если нужен уровень детальнее
        for i in range(n_children):
            for ter_id, ter_class in zip(terr_ids, terr_classes):
                get_children(session, ter_id, ter_class)
            # новый набор для итерации    
            terr_classes = [child for one_class in terr_classes for child in one_class.children]
            # новые id тер-рий
            terr_ids = [one_class.territory_id for one_class in terr_classes]
        
        '''
        '''
        
        
        # all final children in <terr_classes>    
        fin_df = pd.DataFrame([])
        fin_df['territory_id'] = [cl.territory_id for cl in terr_classes]
        fin_df['oktmo'] = [cl.oktmo for cl in terr_classes]
        fin_df['name'] = [cl.name for cl in terr_classes]
        fin_df['geometry'] = [cl.geometry for cl in terr_classes]
        fin_df['centre_point'] = [cl.centre_point for cl in terr_classes]
        
        if (show_level==4)&(fill_in_4):
            # восполняем тем, что есть; на всякий случай сортируем
            # print('Заполнение колонки pop_all с файла towns.geojson')
            towns = gpd.read_file('towns.geojson')
            towns = towns.set_index('territory_id').loc[fin_df.territory_id].reset_index()
            fin_df['pop_all'] = towns[towns.territory_id.isin(fin_df.territory_id)]['population'].values
            fin_df['pop_all'] = fin_df['pop_all'].fillna(0)
            for child in terr_classes:
                child.pop_all = fin_df[fin_df.territory_id==child.territory_id]['pop_all'].values[0]
                
        # у ЛО нет октмо в БД
        with pd.option_context("future.no_silent_downcasting", True):
            fin_df['oktmo'] = fin_df['oktmo'].fillna(0)
        
        # добавляем направления миграции
        if with_mig_dest:
            
            # для уровня областей быстрее из файла (surpise-surprise)
            if from_file & (show_level == 1):
                from_to_geom, from_to_lines = mig_dest_prepared(show_level=show_level, 
                                                                fin_df=fin_df, siblings=[], 
                                                                change_lo_level=change_lo_level, 
                                                                md_year=md_year,
                                                                from_file=from_file)
            else:
                if n_children!=0:
                    siblings = fin_df[['territory_id','oktmo','name','geometry']].copy()
                    
                # если детей не выбирали, т.е одна тер-рия
                if n_children==0:
                    # собираем территории того же родителя, "братьев/сестер"
                    current_territory.parent.territory_type = current_territory.territory_type-1
                    terr_classes = [current_territory.parent]
                    terr_ids = [current_territory.parent.territory_id]
                    
                    
                    for i in range(1):
                        for ter_id, ter_class in zip(terr_ids, terr_classes):
                            get_children(session, ter_id, ter_class)
                        # новый набор для итерации    
                        terr_classes = [child for one_class in terr_classes for child in one_class.children]
                        # новые id тер-рий
                        terr_ids = [one_class.territory_id for one_class in terr_classes]
    
                    siblings = pd.DataFrame([])
                    siblings['territory_id'] = [cl.territory_id for cl in terr_classes]
                    siblings['oktmo'] = [cl.oktmo for cl in terr_classes]
                    siblings['name'] = [cl.name for cl in terr_classes]
                    siblings['geometry'] = [cl.geometry for cl in terr_classes]
                    siblings['centroid'] = [cl.centre_point for cl in terr_classes]
                    show_level = terr_classes[0].territory_type
                
                from_to_geom, from_to_lines = mig_dest_prepared(show_level=show_level, 
                                                                fin_df=fin_df, siblings=siblings,
                                                                change_lo_level=change_lo_level,
                                                                md_year=md_year)
            
        fin_df = main_migration(session, fin_df)    
        fin_df = main_factors(session, fin_df)
        fin_df = gpd.GeoDataFrame(fin_df, geometry='geometry')
        # меняем, чтобы удалось преобразовать в json
        fin_df['centre_point'] = fin_df['centre_point'].astype('str')
        
    if 'oktmo' in fin_df.columns:
        fin_df = fin_df.drop(columns=['oktmo']).fillna(0) 
    
    
    if with_mig_dest:
        return [fin_df, from_to_geom, 
                gpd.GeoDataFrame(from_to_lines, geometry='line')]

    else:
        return fin_df
    

def main_factors(session, fin_df):
    sdf = pd.read_csv(file_path+'superdataset_iqr.csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)
    print(fin_df['oktmo'])
    max_year = np.min([sdf.year.max(), 2022])
    sdf = sdf[(sdf.year==max_year)&(sdf.oktmo.isin(fin_df['oktmo']))]
    sdf = sdf.sort_values(by='year').drop(['name','year', 'saldo'],
                   axis='columns').reset_index(drop=True)
    fin_df = fin_df.merge(sdf, on='oktmo', how='left')
    fin_df = rus_clms_factors(fin_df)
    return fin_df 


def main_migration(session, fin_df):
    df = pd.read_csv(file_path+'in_out_migration_diff_types.csv', index_col=0)[
        ['vozr','migr','municipality','oktmo','year','in_value','out_value']]
    # отбираем только общую миграцию
    df = df[df.migr=='Миграция, всего'].drop(columns=['migr'])
    fin_df[['younger_in','work_in','old_in',
          'younger_out','work_out','old_out']] = 0
    df['oktmo'] = df['oktmo'].astype(str)
    
    oktmos = fin_df['oktmo'].values
    needed_part = df[df.year==df.year.max()]
    needed_part = needed_part[needed_part.oktmo.isin(oktmos)]
    
    # эффективнее сразу для всех, но пока непонятно как
    for oktmo in oktmos:
        check = needed_part[needed_part.oktmo==oktmo]
        # для выбранной тер-рии
        if check.shape[0]:
            # Никольское  -- по 2 строки
            area_part = check.sort_values(by='vozr'
                                         ).drop_duplicates(['vozr','municipality'])

            all_in, all_out, old_in, old_out, \
            work_in, work_out = area_part[['in_value','out_value']].values.flatten()
            young_in, young_out = all_in-old_in-work_in, all_out-old_out-work_out

            fin_df.loc[fin_df.oktmo==oktmo, ['younger_in','work_in','old_in',
              'younger_out','work_out','old_out']] = [young_in,work_in,old_in,
                                                      young_out,work_out,old_out]
            
    fin_df = rus_clms_mig(fin_df, detailed=False)    
    return fin_df 
    

def main_info(session, current_territory, show_level):
    # ____ Узнаем уровень территории
    try:
        url = terr_api + f'api/v1/territory/{current_territory.territory_id}'
        r = session.get(url)
        r_main = r.json()
        
        ter_type = r_main['territory_type']['id']
        # город федерального значения приравниваем к области
        # у СПб тип 18?
        if (ter_type==17) or (ter_type==18):
            ter_type=1
            
        current_territory.territory_type = ter_type
        current_territory.name = r_main['name']
        current_territory.oktmo = r_main['oktmo_code']
        geom_data = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
        current_territory.geometry = geom_data.values[0][0]
        current_territory.centre_point = create_point(r_main['centre_point'])
        
        try:
            current_territory.parent = Territory(r_main['parent']['id'])
            current_territory.pop_all = r_main['properties']['Численность населения']
            current_territory.parent.name = r_main['parent']['name']
        except KeyError:
            current_territory.pop_all = 0
            pass
        
        if show_level == current_territory.territory_type:
            current_territory.df['oktmo'] = current_territory.oktmo
            current_territory.df['geometry'] = geom_data
            current_territory.df['name'] = current_territory.name
            current_territory.df['centre_point'] = current_territory.centre_point
            
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    
def child_to_class(x, parent_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
    child.centre_point = create_point(x['centre_point'])
    child.territory_type = parent_class.territory_type+1
    
    child.parent = parent_class
    parent_class.children.append(child)


def get_children(session, parent_id, parent_class):
    #print(f'children for {parent_id}')
    try:
        url= terr_api + 'api/v2/territories'
        params = {'parent_id':parent_id,'get_all_levels':'false','cities_only':'false','page_size':1000}
        r_u = session.get(url, params=params)
        r = r_u.json()
        if r['results']:
            children_type = parent_class.territory_type+1
            res = pd.json_normalize(r['results'], max_level=0)
            fin = res[['territory_id','name','oktmo_code','centre_point']].copy()
                                   
            if children_type <= 3:
                with_geom = gpd.GeoDataFrame.from_features(r['results']
                                                          ).set_crs(epsg=4326)['geometry']
                fin.loc[:,'geometry'] = with_geom
                                   
            else:
                fin.loc[:,'geometry'] = res['centre_point'].apply(lambda x: create_point(x))
                fin = fin.set_geometry('geometry').set_crs(epsg=4326)
            
            
            fin.apply(child_to_class, parent_class=parent_class, axis=1)
        
    except:
        raise requests.exceptions.RequestException(f'Problem with {r_u.url}')     


def child_to_class_onetry(x, parent_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
    child.territory_type = 4

    child.parent = parent_class
    parent_class.children.append(child)
    

def get_children_one_try(session, parent_class):
    try:
        url = terr_api + 'api/v2/territories'
        params = {'parent_id':parent_class.territory_id,'get_all_levels':'true',
                  'cities_only':'true','page_size':10000}
        r_u = session.get(url, params=params)
        r = r_u.json()
        
        if r['results']:
            res = pd.json_normalize(r['results'], max_level=0)
            fin = res[['territory_id','name','oktmo_code','parent']].copy()
            fin.loc[:,'geometry'] = res['centre_point'].apply(lambda x: create_point(x))
            fin = fin.set_geometry('geometry').set_crs(epsg=4326)
            
            fin.apply(child_to_class_onetry, parent_class=parent_class, axis=1)
    except:
        raise requests.exceptions.RequestException(f'Problem with {r_u.url}')
        
        
def detailed_migration(session, current_territory, fin_df):
    
    df = pd.read_csv(file_path+'in_out_migration_diff_types.csv', index_col=0)[
        ['vozr','migr','municipality','oktmo','year','in_value','out_value']]
    df['oktmo'] = df['oktmo'].astype(str)

    clms = [[f'russia_{dest}',f'outside_region_{dest}',f'inside_region_{dest}',
           f'international_{dest}',f'interregional_{dest}',f'all_mig_{dest}',
           f'other_countries_{dest}',f'cis_counries_{dest}'] for dest in ['in','out']]
    clms = np.array(clms).flatten()
    fin_df[clms] = 0
    
    oktmo = current_territory.oktmo
    needed_part = df[(df.oktmo.isin([oktmo]))&(df.vozr=='Всего')]
    area_part = needed_part.sort_values(by=['year','migr']
                                         ).drop_duplicates(['year','migr','municipality'])
    # чтобы при группировке отсутствующие категории все равно указывались
    area_part.migr = pd.Categorical(area_part.migr, 
                                categories=area_part.migr.unique(), 
                                ordered=True)
    arr = area_part.groupby(['year','migr']
                           ).sum()[['in_value','out_value']].values.flatten()
    
    # shape == (nunique years, nunique migr types * migr.groups)
    arr = arr.reshape(area_part.year.nunique(), area_part.migr.nunique()*2)
    
    uniq_years = area_part.year.unique()
    for year_idx in range(area_part.year.nunique()):
        fin_df.loc[uniq_years[year_idx], clms[:8]] = arr[year_idx, 0::2]
        fin_df.loc[uniq_years[year_idx], clms[8:]] = arr[year_idx, 1::2]
    
    fin_df = rus_clms_mig(fin_df, detailed=True)
    
    return fin_df
    
    
def detailed_factors(session, current_territory, fin_df):
    sdf = pd.read_csv(file_path+'superdataset_iqr.csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)

    oktmo = current_territory.oktmo
    sdf = sdf[(sdf.oktmo.isin([oktmo]))&(sdf.year>=2019)]
    sdf = sdf.sort_values(by='year').drop(['name', 'saldo'],
                       axis='columns').reset_index(drop=True)
    
    fin_df = fin_df.merge(sdf, how='left', on='year')
    fin_df = rus_clms_factors(fin_df)
    fin_df = fin_df.rename(columns = {'year':'Год'})
    return fin_df


# _________ Направления миграции _________

def change_children_to_parents(result, dict_lo, uniq_ids):
    dict_lo_geom = dict_lo['geometry_parent']
    result.loc[~result['from_territory_id'].isin([*uniq_ids, 0]), 
               'from_geometry'] = result[~result['from_territory_id'].isin([*uniq_ids, 0])
                                            ]['from_territory_id'
                                             ].apply(lambda x: dict_lo_geom[x])
    result.loc[~result['to_territory_id'].isin([*uniq_ids, 0]), 
               'to_geometry'] = result[~result['to_territory_id'].isin([*uniq_ids, 0])
                                            ]['to_territory_id'
                                             ].apply(lambda x: dict_lo_geom[x])

    dict_lo_centroid = dict_lo['centroid']
    result.loc[~result['from_territory_id'].isin([*uniq_ids, 0]), 
               'from_centroid'] = result[~result['from_territory_id'].isin([*uniq_ids, 0])
                                            ]['from_territory_id'
                                             ].apply(lambda x: dict_lo_centroid[x])
    result.loc[~result['to_territory_id'].isin([*uniq_ids, 0]), 
               'to_centroid'] = result[~result['to_territory_id'].isin([*uniq_ids, 0])
                                            ]['to_territory_id'
                                             ].apply(lambda x: dict_lo_centroid[x])

    dict_lo_parents = dict_lo['territory_id_parent']
    result.loc[~result['from_territory_id'].isin([*uniq_ids, 0]), 
               'from_territory_id'] = result[~result['from_territory_id'].isin([*uniq_ids, 0])
                                            ]['from_territory_id'
                                             ].apply(lambda x: dict_lo_parents[x])
    result.loc[~result['to_territory_id'].isin([*uniq_ids, 0]), 
               'to_territory_id'] = result[~result['to_territory_id'].isin([*uniq_ids, 0])
                                            ]['to_territory_id'
                                             ].apply(lambda x: dict_lo_parents[x])
    return result


def mig_dest_multipolygons(result, fin_df_with_centre):
    # собираем в формат: полигон, сколько приехало, сколько уехало
    from_summed = result.groupby('from_territory_id'
                            )['n_people'].sum().reset_index(
                                ).rename(columns={'n_people':'people_out',
                                                  'from_territory_id':'territory_id'})
    to_summed = result.groupby('to_territory_id'
                              )['n_people'].sum().reset_index(
                                ).rename(columns={'n_people':'people_in',
                                                 'to_territory_id':'territory_id'})
    from_to_summed = from_summed.merge(to_summed, how='outer').fillna(0)
    # добавляем geometry
    from_to_geom = from_to_summed.merge(fin_df_with_centre[['territory_id','name','geometry'
                                                       ]], how='left')
    from_to_geom.territory_id = from_to_geom.territory_id.astype(int)
    from_to_geom.loc[1:, 'geometry'] = gpd.GeoSeries.from_wkt(from_to_geom['geometry'
                                                           ].iloc[1:].astype(str))
    from_to_geom = gpd.GeoDataFrame(from_to_geom, geometry='geometry')
    
    # собираем в формат: линия, октуда, куда, сколько людей
    # longitude, latitude
    outer_point = gpd.points_from_xy([37.633400192402426],
                                     [55.75918327956314])[0]
    result.loc[result['from_territory_id']==0, 'from_centroid'] = outer_point
    result.loc[result['to_territory_id']==0, 'to_centroid'] = outer_point
    result.loc[:,'from_centroid'] = gpd.GeoSeries.from_wkt(result['from_centroid'].astype(str))
    result.loc[:,'to_centroid'] = gpd.GeoSeries.from_wkt(result['to_centroid'].astype(str))

    result.loc[:,'line'] = result.apply(lambda x: LineString([x['from_centroid'], 
                                         x['to_centroid']]), axis=1)
    
    
    from_to_lines = result[['from_territory_id','to_territory_id','n_people','line']]
    from_to_lines.loc[:,'from_territory_id'] = from_to_lines.from_territory_id.astype(int)
    from_to_lines.loc[:,'to_territory_id'] = from_to_lines.to_territory_id.astype(int)
    return [from_to_geom, from_to_lines]


def mig_dest_prepared(show_level, fin_df, siblings, 
                      change_lo_level=False, md_year=2022, from_file=False):
    
    
    if show_level <= 1:
        if from_file:
            siblings = pd.read_csv(file_path+'bd_id_geom_regions.csv', index_col=0)
            
        
        df_with_geom = siblings.copy()
        res_years = pd.read_csv(file_path+'obl_mig_no_tid_19-23.csv', 
                                index_col=0)
        yeartab = siblings.drop(columns=['oktmo']
                               ).merge(res_years, on='name')
        # при ЛО где-то дублируется
        if 'territory_id_x' in yeartab.columns:
           yeartab = yeartab.rename(columns={'territory_id_x':'territory_id'}
                                    ).drop(columns=['territory_id_y'])
        
        
        result = create_graph(yeartab, md_year)
        
    else:
        df_with_geom = pd.read_csv(file_path+'lo_3_parents.csv', 
                                   index_col=0)
        if show_level > 2:
            # result = pd.read_csv(file_path+f'graph_LO_{show_level}level_{md_year}.csv',index_col=0)
            res = pd.read_csv(file_path + 'graph_LO_3_no_spb_19-22.csv', index_col=0)
            result = res[res.year==md_year].drop(columns=['year']) # для /main_info/ всегда посл.год

        else:
            pre_result = pd.read_csv(file_path+'for_graph_LO_2_no_spb.csv',
                                     index_col=0)
            
            result = create_graph(pre_result, md_year)
            
    # линии только из заданных территорий
    uniq_ids = fin_df.territory_id.unique()
    result = result[(result.from_territory_id.isin([*uniq_ids])
                    )|(result.to_territory_id.isin([*uniq_ids]))
                   ]
    
    if show_level==2:
        # взять только колонки _parent и поменять названия
        df_with_geom = df_with_geom.iloc[:,4:].drop_duplicates()
        df_with_geom.columns = ['territory_id','oktmo','name','geometry']
    
    if show_level>1:
        df_with_geom.loc[:,'geometry'] = gpd.GeoSeries.from_wkt(df_with_geom['geometry'].astype(str))
        df_with_geom.loc[:,'centroid'] = df_with_geom.geometry.apply(lambda x: x.centroid)

    
    # мерджим, чтобы добавить инф-ию по id и geometry
    #  для узла "откуда"
    result = result.merge(df_with_geom[['territory_id','name','geometry','centroid']], 
                      left_on='from_territory_id', 
                      right_on='territory_id', how='left'
                      ).rename(columns={'geometry':'from_geometry',
                                       'centroid':'from_centroid'}
                              ).drop(columns=['territory_id'])
    #  для узла "куда"
    result = result.merge(df_with_geom[['territory_id','name','geometry','centroid']], 
                      left_on='to_territory_id', 
                      right_on='territory_id', how='left'
                         ).rename(columns={'geometry':'to_geometry',
                                       'centroid':'to_centroid'}
                                 ).drop(columns=['territory_id'])
    
    # сгруппировать побочные территории, чтобы не нагружать рисунок
    # если при подъеме будет не ЛО, если задали подъем, если это не detailed=True 
    if (show_level-1 > 1)&(change_lo_level):
        lo_for_dict = df_with_geom[['territory_id','territory_id_parent',
                    'geometry_parent']].copy()
        lo_for_dict.loc[:,'geometry_parent'] = gpd.GeoSeries.from_wkt(
            lo_for_dict['geometry_parent'].astype(str))
        lo_for_dict.loc[:,'centroid'] = lo_for_dict.geometry_parent.apply(lambda x: x.centroid)
        dict_lo = lo_for_dict.set_index('territory_id').to_dict()
        
        # все территории, не относящиеся к братьям/сестрам, группируем на уровень выше
        uniq_ids = siblings.territory_id.unique()
        result = change_children_to_parents(result, dict_lo, uniq_ids)
        
        # заменяем наш дф с геометрией детей-родителей;
        # для нерассматриваемых id заменяем на территорию выше
        bool_idx = ~df_with_geom.territory_id.isin(uniq_ids)
        df_with_geom.loc[bool_idx, df_with_geom.columns[:4]
                        ] = df_with_geom.loc[bool_idx, df_with_geom.columns[4:-1]].values
        df_with_geom = df_with_geom.drop_duplicates()
                
    from_to_geom, from_to_lines = mig_dest_multipolygons(result, df_with_geom)
    from_to_geom = from_to_geom.drop_duplicates()
    
    # колонки на русском
    from_to_geom = from_to_geom.rename(columns={'people_in':'Количество приехавших',
                                                'people_out':'Количество уехавших',
                                                'name':'Название территории'})
    
    from_to_lines = from_to_lines.drop_duplicates()
    from_to_lines.loc[:,'to_territory_id'] = from_to_lines['to_territory_id'].astype(int)
    from_to_lines.loc[:,'from_territory_id'] = from_to_lines['from_territory_id'].astype(int)

    from_to_geom = from_to_geom.rename(columns={'n_people':'Количество уехавших',
                                                'people_out':'Количество уехавших',
                                                'name':'Название территории'})
    
    return from_to_geom, from_to_lines



def create_graph(pre_result, md_year):
    yeartab = pre_result[pre_result.year==md_year].copy()

    yeartab['nInt_in'] = yeartab['Int_in']/yeartab['Int_in'].sum()
    yeartab['nInt_out'] = yeartab['Int_out']/yeartab['Int_out'].sum()
    yeartab['saldo'] = yeartab['Int_in']-yeartab['Int_out']+yeartab['Ext_in']-yeartab['Ext_out']

    w=lambda u, v: round((yeartab.loc[yeartab.territory_id==u, 
                                      'Int_out'].values*yeartab.loc[yeartab.territory_id==v, 
                                                                    'nInt_in'].values)[0] , 0)

    edg0 = np.array([(int(u), int(v), w(u,v)) for u, v in permutations(yeartab.territory_id, 
                                                                       2) if w(u,v)>0])
    edg1 = np.array([(0, int(v), yeartab.loc[yeartab.territory_id==v, 
                                             'Ext_in'].values[0]) for v in yeartab.territory_id ])
    edg2 = np.array([(int(u), 0, yeartab.loc[yeartab.territory_id==u, 
                                             'Ext_out'].values[0]) for u in yeartab.territory_id ])

    dd = pd.concat([pd.DataFrame(edg0),
                    pd.DataFrame(edg1),
                    pd.DataFrame(edg2)]).astype(int)
    dd.columns=['from_territory_id','to_territory_id','n_people']
    
    return dd


# _________ Для русских колонок _________

def rus_clms_factors(fin_df):
    mig_f_rus = ['Численность населения (чел.)','Среднее число работников организаций (чел.)', 'Средняя зарплата (руб.)', 
                 'Площадь торговых залов магазинов (кв. м.)', 'Количество мест в ресторанах кафе барах (место)', 
                 'Оборот розничной торговли без малых предприятий (тыс. руб.)','Оборот общественного питания (тыс. руб.)', 
                 'Введенные жилые дома (кв. м)', 'Введенные квартиры (шт. на 1000 чел.)',
                 'Жилая площадь на одного человека (кв. м.)', 'Число спортивных сооружений (шт.)',
                 'Объекты бытового обслуживания (шт.)','Длина дорог (км)', 'Поголовье скота всех видов (шт.)', 
                 'Урожайность овощей (цент.)',  'Продукция сельского хозяйства (тыс. руб.)',
                 'Инвестиции в основной капитал (тыс. руб.)', 'Доходы бюджета (тыс. руб.)', 
                 'Износ основного фонда (тыс. руб.)','Число музеев (шт.)', 'Число парков культуры (шт.)', 
                 'Число театров (шт.)', 'Лечебно-профилактические организации (шт.)','Мощность поликлиник (шт.)',
                 'Число мест в дошкольных обр. учреждениях (шт.)', 'Число общеобразовательных организаций (шт.)',
                 'Затраты на охрану окружающей среды (тыс. руб.)', 'Отгружено товаров собственного производства (тыс. руб.)']
    mig_f_eng = ['popsize','avgemployers', 'avgsalary', 'shoparea',
                   'foodseats', 'retailturnover', 'foodservturnover', 'consnewareas',
                   'consnewapt', 'livarea', 'sportsvenue', 'servicesnum', 'roadslen',
                   'livestock', 'harvest', 'agrprod', 'invest', 'budincome', 'funds',
                   'museums', 'parks', 'theatres', 'hospitals', 'cliniccap',
                   'beforeschool', 'schoolnum', 'naturesecure', 'factoriescap']
    
    return fin_df.rename(columns = dict(zip(mig_f_eng,mig_f_rus)))


def rus_clms_mig(fin_df, detailed=False):
    if detailed:
        mig_dest_rus = ['В пределах России', 'Внешняя (для региона) миграция', 
                        'Внутрирегиональная', 'Международная', 'Межрегиональная', 
                        'Миграция всего', 'С другими зарубежными странами', 'Со странами СНГ']
        mig_dest_eng = ['russia', 'outside_region', 'inside_region',
                       'international', 'interregional', 'all_mig',
                       'other_countries', 'cis_counries']
        rus_clms = [f'{dest}{mig}' for dest in ['Входящая. ',
                                                'Исходящая. '] for mig in mig_dest_rus]
        eng_clms = [f'{mig}_{dest}' for dest in ['in','out'] for mig in mig_dest_eng]
    
    else:
        eng_clms = ['younger_in', 'work_in', 'old_in', 
                    'younger_out', 'work_out', 'old_out']
        rus_clms = ['Входящая. Моложе трудоспособного возраста','Входящая. Трудоспособного возраста',
                     'Входящая. Старше трудоспособного возраста',
                   'Исходящая. Моложе трудоспособного возраста','Исходящая. Трудоспособного возраста',
                     'Исходящая. Старше трудоспособного возраста']
    
    return fin_df.rename(columns = dict(zip(eng_clms,rus_clms)))