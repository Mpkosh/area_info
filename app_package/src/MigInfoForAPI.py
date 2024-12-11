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
import networkx as nx
from itertools import permutations


social_api = os.environ.get('SOCIAL_API')
terr_api = os.environ.get('TERRITORY_API') 
file_path = 'app_package/src/'

       
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
        raise ValueError(f'Show level (given: {show_level}) must be >= territory type (given: {current_territory.territory_type})')

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
    
    return fin_df


def info(territory_id, show_level=0, detailed=False, 
         with_mig_dest=False, fill_in_4=False):
    md_year = 2022
    session = requests.Session()
    current_territory = Territory(territory_id)
    main_info(session, current_territory, show_level)
    
    if detailed:
        #show_level = current_territory.territory_type
        fin_df = pd.DataFrame([])
        fin_df = detailed_migration(session, current_territory, fin_df)
        fin_df = fin_df.rename_axis('year').reset_index()
        fin_df = detailed_factors(session, current_territory, fin_df)
        
    else:
        if show_level < current_territory.territory_type:
            raise ValueError(f'Show level (given: {show_level}) must be >= territory type (given: {current_territory.territory_type})')

        n_children = show_level - current_territory.territory_type
        terr_classes = [current_territory]
        terr_ids = current_territory.df['territory_id'].values
        
        
        if show_level == 4:
            get_children_one_try(session, current_territory)
            terr_classes = current_territory.children
            terr_ids = [one_class.territory_id for one_class in terr_classes]
            
        else:    
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
        
        if (show_level==4)&(fill_in_4):
            # восполняем тем, что есть; на всякий случай сортируем
            # print('Заполнение колонки pop_all с файла towns.geojson')
            towns = gpd.read_file(file_path+'towns.geojson')
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
            # для ЛО
            lo2 = info_for_graph(1, 2, md_year)
            lo2 = lo2[lo2.territory_id!=territory_id]

            fin = pd.concat([lo2,fin_df[['geometry','territory_id',
                                              'name','oktmo']]])
            from_to_geom, from_to_lines = mig_destinations_df(fin[['geometry',
                                                                   'territory_id',
                                                                   'name','oktmo']], 
                                                              md_year,
                                                              fin_df.territory_id.unique())
            from_to_lines['to_territory_id'] = from_to_lines['to_territory_id'].astype(int)
        
        fin_df = main_migration(session, fin_df)    
        fin_df = main_factors(session, fin_df)
        
    if 'oktmo' in fin_df.columns:
        fin_df = fin_df.drop(columns=['oktmo']).fillna(0) 
            
    if with_mig_dest:
        return [fin_df, from_to_geom, from_to_lines]

    else:
        return fin_df
    

def main_factors(session, fin_df):
    sdf = pd.read_csv(file_path+'superdataset (full data).csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)
    sdf = sdf[(sdf.year==sdf.year.max())&(sdf.oktmo.isin(fin_df['oktmo']))]
    sdf = sdf.sort_values(by='year').drop(['name','year', 'saldo', 'popsize'],
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
        current_territory.territory_type = r_main['territory_type']['territory_type_id']
        current_territory.name = r_main['name']
        current_territory.oktmo = r_main['oktmo_code']
        geom_data = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
        current_territory.geometry = geom_data.values[0][0]
        
        if show_level == current_territory.territory_type:
            current_territory.df['oktmo'] = current_territory.oktmo
            current_territory.df['geometry'] = geom_data
            current_territory.df['name'] = current_territory.name
            
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
def child_to_class(x, parent_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
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
            fin = res[['territory_id','name','oktmo_code']].copy()
                                   
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
        params = {'parent_id':parent_class.territory_id,'get_all_levels':'true','cities_only':'true','page_size':10000}
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
    sdf = pd.read_csv(file_path+'superdataset (full data).csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)

    oktmo = current_territory.oktmo
    sdf = sdf[(sdf.oktmo.isin([oktmo]))&(sdf.year>=2019)]
    sdf = sdf.sort_values(by='year').drop(['name', 'saldo', 'popsize'],
                       axis='columns').reset_index(drop=True)
    
    fin_df = fin_df.merge(sdf, how='left', on='year')
    fin_df = rus_clms_factors(fin_df)
    fin_df = fin_df.rename(columns = {'year':'Год'})
    return fin_df


# _________ Направления миграции _________

def mig_destinations(fin_df):
    fin_df = fin_df.copy()
    # широта и долгота центральной точки
    fin_df['centroid'] = fin_df.geometry.apply(lambda x: x.centroid)
    fin_df[['latitude_dd','longitude_dd']] = fin_df.apply(lambda x: (x['centroid'].x, x['centroid'].y), 
                                                          axis=1, result_type='expand')
    
    ddf = pd.read_csv(file_path+'in_out_migration_diff_types.csv', index_col=0)[
        ['vozr','migr','municipality','oktmo','year','in_value','out_value']]
    ddf['oktmo'] = ddf['oktmo'].astype(str)
    oktmos = fin_df['oktmo'].values
    
    # отбираем по ОКТМО нужные районы
    needed_part = ddf[(ddf.oktmo.isin(oktmos))&
                      (ddf.vozr=='Всего')&
                      (ddf.migr.isin(['Миграция, всего', 'Внутрирегиональная']))]
    # добавляем territory_id и используем только его как id. 
    needed_part = needed_part.merge(fin_df[['territory_id','oktmo']], 
                      how='right', on='oktmo')
    area_part = needed_part.sort_values(by=['year','migr','territory_id']
                                       ).drop_duplicates(['year','migr',
                                                          'territory_id'])
    # собираем значения
    arr = area_part.groupby(['territory_id','year','migr']
                           ).sum()[['in_value','out_value']].reset_index()
    
    # отдельно для внутрирег. и для общей миграции
    q_region = arr[arr.migr=='Внутрирегиональная'].drop(columns=['migr'])
    q_region = q_region.rename(columns={'in_value':'Int_in',
                                        'out_value':'Int_out'})
    q_all = arr[arr.migr=='Миграция, всего'].drop(columns=['migr'])
    # объединяем
    q_fin = q_region.merge(q_all)
    
    q_fin['Ext_in'] = q_fin["in_value"]-q_fin["Int_in"]
    q_fin['Ext_out'] = q_fin["out_value"]-q_fin["Int_out"]
    q_fin.drop(columns=['in_value','out_value'])

    q_fin = q_fin[['territory_id','year','Int_in','Int_out','Ext_in','Ext_out']
                   ].merge(fin_df[['territory_id','name','latitude_dd','longitude_dd']], 
                           how='right', on='territory_id')
    return fin_df, q_fin


# функция от А.Н.
def mig_destinations_df(fin_df, md_year=2022, uniq_ids=[]):
    fin_df_with_centre, q_fin = mig_destinations(fin_df)
    
    # данные по заданному году; нормируем
    yeartab = q_fin[q_fin.year==md_year].copy()
    yeartab['nInt_in'] = yeartab['Int_in']/yeartab['Int_in'].sum()
    yeartab['nInt_out'] = yeartab['Int_out']/yeartab['Int_out'].sum()
    yeartab['saldo'] = yeartab['Int_in']-yeartab['Int_out']+yeartab['Ext_in']-yeartab['Ext_out']
    
    # строим граф
    G = nx.DiGraph()
    G.add_nodes_from(zip(yeartab.territory_id.unique(),
                          yeartab[['saldo','name', 'latitude_dd', 'longitude_dd']
                                 ].to_dict(orient= 'records'))
                    )
    w = lambda u, v: round((yeartab.loc[yeartab.territory_id==u, 'Int_out'
                                       ].values * yeartab.loc[yeartab.territory_id==v, 
                                                              'nInt_in'].values)[0], 
                           0)
    edg = np.array([(int(u), int(v), w(u,v)) for u, v in permutations(yeartab.territory_id, 2
                                                                     ) if w(u,v)>0])
    G.add_weighted_edges_from(edg)
    # для баланса добавляем внешний узел
    outer_node_id = 0
    G.add_node(outer_node_id, name='External', saldo = -yeartab.saldo.sum(), 
                                latitude_dd = yeartab.latitude_dd.min()-.5, 
                                longitude_dd = yeartab.longitude_dd.mean()) 
    edg = np.array([(outer_node_id, int(v), yeartab.loc[yeartab.territory_id==v, 'Ext_in'].values[0]
                    ) for v in yeartab.territory_id ])
    G.add_weighted_edges_from(edg)
    edg = np.array([(int(u), outer_node_id, yeartab.loc[yeartab.territory_id==u, 'Ext_out'].values[0]
                    ) for u in yeartab.territory_id ])
    G.add_weighted_edges_from(edg)
    # все в датафрейм
    ndf = nx.to_pandas_edgelist(G).rename(columns={'source':'from_territory_id',
                                                   'target':'to_territory_id',
                                                   'weight':'n_people'})
    
    # мерджим, чтобы добавить инф-ию по id и geometry
    #  для узла "откуда"
    result = ndf.merge(fin_df_with_centre[['territory_id','geometry','centroid']], 
                      left_on='from_territory_id', 
                      right_on='territory_id', how='left'
                      ).rename(columns={'geometry':'from_geometry',
                                       'centroid':'from_centroid'}
                              ).drop(columns=['territory_id'])
    #  для узла "куда"
    result = result.merge(fin_df_with_centre[['territory_id','geometry','centroid']], 
                      left_on='to_territory_id', 
                      right_on='territory_id', how='left'
                         ).rename(columns={'geometry':'to_geometry',
                                       'centroid':'to_centroid'}
                                 ).drop(columns=['territory_id'])
    # убираем линии миграции у вспомогательных районов
    result = result[(result.from_territory_id.isin(uniq_ids)
                    )|(result.to_territory_id.isin(uniq_ids))
                   ]
    from_to_geom, from_to_lines = mig_dest_multipolygons(result, fin_df_with_centre)
    return from_to_geom, from_to_lines


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
    from_to_geom = from_to_summed.merge(fin_df_with_centre[['territory_id','geometry'
                                                       ]], how='left')
    from_to_geom.territory_id = from_to_geom.territory_id.astype(int)
    #.to_json()
    
    # собираем в формат: линия, октуда, куда, сколько людей
    # longitude, latitude
    outer_point = gpd.points_from_xy([37.633400192402426],
                                     [55.75918327956314])[0]
    result.loc[result['from_territory_id']==0, 'from_centroid'] = outer_point
    result.loc[result['to_territory_id']==0, 'to_centroid'] = outer_point
    result['line'] = result.apply(lambda x: LineString([x['from_centroid'], 
                                         x['to_centroid']]), axis=1)

    from_to_lines = result[['from_territory_id','to_territory_id','n_people','line']]
    from_to_lines.loc[:,'from_territory_id'] = from_to_lines.from_territory_id.astype(int)
    from_to_lines.loc[:,'to_territory_id'] = from_to_lines.to_territory_id.astype(int)
    #from_to_lines = gpd.GeoDataFrame(from_to_lines, geometry='line')#.to_json()
    
    return [from_to_geom, from_to_lines]



# _________ Для русских колонок _________

def rus_clms_factors(fin_df):
    mig_f_rus = ['Среднее число работников организаций (чел.)', 'Средняя зарплата (руб.)', 
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
    mig_f_eng = ['avgemployers', 'avgsalary', 'shoparea',
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