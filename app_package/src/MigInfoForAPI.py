# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:21:24 2024

@author: user
"""

import requests
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os


social_api = os.environ.get('SOCIAL_API')
terr_api = os.environ.get('TERRITORY_API') 

       
class Territory():
    
    def __init__(self, territory_id=34):
        self.territory_id = territory_id
        self.df = pd.DataFrame([])
        self.df['territory_id'] = [self.territory_id]
        self.children = []
        self.parent = 0
        

def create_point(x):
    return Point(x['coordinates'])       


def info(territory_id, show_level=0, detailed=False):
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

        # у ЛО нет октмо в БД
        with pd.option_context("future.no_silent_downcasting", True):
            fin_df['oktmo'] = fin_df['oktmo'].fillna(0)
        fin_df = main_migration(session, fin_df)    
        fin_df = main_factors(session, fin_df)
        
        #fin_df = fin_df.rename(columns = {''}
        
    return fin_df.drop(columns=['oktmo']).fillna(0)
    

def main_factors(session, fin_df):
    sdf = pd.read_csv('app_package/src/superdataset (full data).csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)
    sdf = sdf[(sdf.year==sdf.year.max())&(sdf.oktmo.isin(fin_df['oktmo']))]
    sdf = sdf.sort_values(by='year').drop(['name','year', 'saldo', 'popsize'],
                   axis='columns').reset_index(drop=True)
    fin_df = fin_df.merge(sdf, on='oktmo', how='left')
    fin_df = rus_clms_factors(fin_df)
    return fin_df 



def main_migration(session, fin_df):
    df = pd.read_csv('app_package/src/in_out_migration_diff_types.csv', index_col=0)[
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
    url = terr_api + f'api/v1/territory/{current_territory.territory_id}'
    r_main = session.get(url).json()
    
    current_territory.territory_type = r_main['territory_type']['territory_type_id']
    current_territory.name = r_main['name']
    current_territory.oktmo = r_main['oktmo_code']
    geom_data = gpd.GeoDataFrame.from_features([r_main])[['geometry']]
    current_territory.geometry = geom_data.values[0][0]
    
    if show_level == current_territory.territory_type:
        current_territory.df['oktmo'] = current_territory.oktmo
        current_territory.df['geometry'] = geom_data
        current_territory.df['name'] = current_territory.name

    
def child_to_class(x, ter_class):
    child = Territory(x['territory_id'])
    child.name = x['name']
    child.oktmo = x['oktmo_code']
    child.df['name'] = child.name
    child.geometry = x['geometry']
    child.territory_type = ter_class.territory_type+1
    
    child.parent = ter_class
    ter_class.children.append(child)


def get_children(session, parent_id, ter_class):
    
    url= terr_api + f'api/v2/territories?parent_id={parent_id}&get_all_levels=false&cities_only=false&page_size=1000'
    r = session.get(url).json()

    if r['results']:
        children_type = ter_class.territory_type+1
        res = pd.json_normalize(r['results'], max_level=0)
        fin = res[['territory_id','name','oktmo_code']].copy()
                               
        if children_type <= 3:
            with_geom = gpd.GeoDataFrame.from_features(r['results']
                                                      ).set_crs(epsg=4326)['geometry']
            fin.loc[:,'geometry'] = with_geom
                               
        else:
            fin.loc[:,'geometry'] = res['centre_point'].apply(lambda x: create_point(x))
            fin = fin.set_geometry('geometry').set_crs(epsg=4326)
        
        
        fin.apply(child_to_class, ter_class=ter_class, axis=1)
        
        
def detailed_migration(session, current_territory, fin_df):
    
    df = pd.read_csv('app_package/src/in_out_migration_diff_types.csv', index_col=0)[
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
    sdf = pd.read_csv('app_package/src/superdataset (full data).csv')
    sdf['oktmo'] = sdf['oktmo'].astype(str)

    oktmo = current_territory.oktmo
    sdf = sdf[(sdf.oktmo.isin([oktmo]))&(sdf.year>=2019)]
    sdf = sdf.sort_values(by='year').drop(['name', 'saldo', 'popsize'],
                       axis='columns').reset_index(drop=True)
    
    fin_df = fin_df.merge(sdf, how='left', on='year')
    fin_df = rus_clms_factors(fin_df)
    fin_df = fin_df.rename(columns = {'year':'Год'})
    return fin_df


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