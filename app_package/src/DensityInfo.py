import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import shape
import numpy as np
from app_package.src import AreaOnMapFile
import os
from shapely.geometry import Point

social_api = os.environ.get('SOCIAL_API')
territories_api = os.environ.get('TERRITORY_API')

 
def create_point(x):
    return Point(x['coordinates'])


def get_all_children_data(session, territory_id=34, from_api=True):
    try:
        # все населенные пункты, ГП, СП, входящие в заданный район
        url = territories_api + 'api/v2/territories'
        params = {'parent_id':territory_id,'get_all_levels':'true','cities_only':'false','page_size':'5000'}
        #url = territories_api + f'api/v1/all_territories?parent_id={territory_id}&get_all_levels=True'
        r = session.get(url, params=params)
        children = pd.DataFrame(r.json())
        # раскрываем json
        res = pd.json_normalize(children['results'], max_level=0)
        res_vill = pd.concat([res.drop('parent', axis='columns'), 
                              pd.json_normalize(res.parent).add_prefix('parent.')],
                              axis=1)
        # меняем тип колонки с геометрией
        for_use = res_vill.copy()
        all_children = gpd.GeoDataFrame(res_vill, 
                             geometry=[shape(d) for d in for_use.pop("geometry")])
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
        
    return all_children


def clip_all_children(all_children, p_id=34):
    p_id = int(p_id)
    
    a = all_children.copy()
    a['centre_point'] = a['centre_point'].apply(lambda x: create_point(x))
    a = a.drop(columns='geometry').set_geometry('centre_point')

    gpsp_with_kids = a[a['parent.id']!=int(p_id)].dissolve(by='parent.id')
    
    # для ЛенОбласти будет непустое
    gpsp_with_parent = a[(a.level==4)&(a['parent.id']!=int(p_id))][['territory_id','name',
                                                           'level','parent.id','parent.name']]

    # если после удаления родителя разные уровни, то там минимум иерархия из двух уровней
    # т.е. можно объединять точки последнего уровня на основе уровня выше 
    if a[a['parent.id']!=p_id].level.unique().shape[0]>1:

        buff = gpsp_with_kids[['centre_point']].merge(gpsp_with_parent,
                                      left_index=True, 
                                      right_on='territory_id'
                                      ).dissolve(by='parent.id')
    else:
        buff = gpsp_with_kids

    return buff


def get_first_children_data(session, territory_id=34, from_api=True):
    try:
        # 34 -- Всеволожский муниципальный район
        url= territories_api + 'api/v1/territory/indicator_values'
        params = {'parent_id':territory_id,'indicator_ids':'1','last_only':'false'}
        r = session.get(url, params=params)
        first_children = pd.DataFrame(r.json())
        
        # df: type, geometry, properties(territory_id,name,...)
        vills_with_geom = pd.json_normalize(first_children['features'], max_level=0)
        # df: territory_id, name, indicators
        vills_with_pop = pd.json_normalize(vills_with_geom['properties'], max_level=0)
    
        if vills_with_pop['indicators'].str.len().min() > 0:
            # раскрываем json с данными о населении
            # на выходе pd Series [[years, pop_values],...]
            pop_vals = vills_with_pop['indicators'].apply(lambda x: pd.json_normalize(x)[['date_value','value']].values)
            # берем года для будущих названий колонок
            clms = pop_vals[0][:,0]
            clms = [c[:4] for c in clms] # ['2019', '2020', '2021', '2022', '2023']
            # собираем все значения в один массив
            # 19 x 5 x -1
            pop_vals_one_arr = np.concatenate(pop_vals).reshape(vills_with_geom.shape[0], len(clms), -1) 
            vills_with_pop[clms] = pop_vals_one_arr[:,:,1]
        vills_with_pop = vills_with_pop.drop('indicators', axis='columns')
        
        for_use = vills_with_geom.copy()
        first_children_f = gpd.GeoDataFrame(vills_with_pop, 
                                           geometry=[shape(d) for d in for_use.pop("geometry")])
        first_children_f['geometry'].crs = 'EPSG:4326'
    except:
        raise requests.exceptions.RequestException(f'Problem with {r.url}')
    
    return first_children_f


def get_color_list():
    # ставим в соответствие цвет для карты
    '''
    alpha = 0.55
    labels_ordered = ['0 — 10','10 — 100','100 — 500','500 — 1 000',
                          '1 000 — 5 000','5 000 — ...']
    color_list = [f'rgba{color[3:-1]}, {alpha})' for color in 
                                   # px.colors.sample_colorscale("viridis", 6)]
    # чтобы сохранить соответствие цвета и категории у всех районов
    common_color_list  = dict(zip(labels_ordered, color_list))
    '''
    common_color_list = {'0 — 10': 'rgba(68, 1, 84, 0.55)',
                         '10 — 100': 'rgba(64, 66, 134, 0.55)',
                         '100 — 500': 'rgba(42, 120, 142, 0.55)',
                         '500 — 1 000': 'rgba(40, 168, 131, 0.55)',
                         '1 000 — 5 000': 'rgba(124, 209, 79, 0.55)',
                         '5 000 — ...': 'rgba(253, 231, 37, 0.55)'}

    return common_color_list


def get_density_data(session, territory_id=34, from_api=False):
    #    Данные о населении по годам в ГП/СП
    first_children_f = get_first_children_data(session, territory_id, from_api)
    #    Плотность населения о ГП/СП
    d_with_dnst = AreaOnMapFile.calculate_density(first_children_f)
    df = d_with_dnst.copy()
    
    # ставим интервалы и цвета для легенды
    labels_ordered = ['0 — 10','10 — 100','100 — 500','500 — 1 000',
                  '1 000 — 5 000','5 000 — ...']
    
    # колонки с плотностью
    dnst_cols = df.columns[df.columns.str.endswith('_dnst')]
    # к колонкам с годом добавляем "_dnst_binned"
    binned_cols = df.columns[df.columns.str.isdecimal()] +'_dnst_binned'
    # добавляем интервалы плотности на каждый год
    df[binned_cols] = df[dnst_cols].apply(pd.cut, bins=[0,10,100,500,1000,5000,100000], 
                                          labels=labels_ordered)

    #    Данные о всех населенных пунктах
    all_children = get_all_children_data(session, territory_id, from_api)
    
    
    # если нет делений меньше (например, работаем с городом, и он не делится дальше)
    if set(df.territory_id.values) == set(all_children.territory_id.values):
        fin_vills_full = df.merge(all_children[['territory_id','parent.id','centre_point'
                                               ]], on='territory_id')
        fin_vills_full['centre_point'] = fin_vills_full['centre_point'
                                                        ].apply(lambda x: create_point(x))
    else:
        # вручную режем по границе КАЖДОГО ГП/СП
        np_clipped = clip_all_children(all_children, p_id=territory_id)
    
        vills_in_area = np_clipped.reset_index().set_crs(epsg=4326
                                                        )[['parent.id','centre_point']]
        fin_vills_full = df.merge(vills_in_area, left_on='territory_id', 
                                  right_on='parent.id')
    
    return df, fin_vills_full, all_children

    
def density_data_geojson(session, territory_id=34, from_api=False):
    # данные о ГП/СП и НП
    _, fin_vills_full, _ = get_density_data(session, territory_id=territory_id, 
                                            from_api=from_api)
    
    full_df = fin_vills_full.drop(columns=['parent.id']
                                  ).rename(columns={'geometry':'geometry_areas',
                                                    'centre_point':'geometry_villages'}
                                                    ).set_geometry('geometry_areas')#.set_index('territory_id')
    # меняем, чтобы удалось преобразовать в geojson
    full_df['geometry_villages'] = full_df['geometry_villages'].astype('str')
    
    return full_df
    