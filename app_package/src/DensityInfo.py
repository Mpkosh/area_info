import geopandas as gpd
import requests
import pandas as pd
from shapely.geometry import shape
import numpy as np
#import plotly.express as px
from app_package.src import AreaOnMapFile


def get_all_children_data(session, territory_id=34, from_api=False):
    # все населенные пункты, ГП, СП, входящие в заданный район
    if from_api:
        get_all_levels_str = 'true'
        url = f'http://10.32.1.107:5300/api/v2/territories?parent_id={territory_id}&get_all_levels={get_all_levels_str}&size=1000'
        r = session.get(url)
        children = pd.DataFrame(r.json())
        #children.to_json(f'json_api_for_map/children_{territory_id}.json')
    else:
        children = pd.read_json(f'json_api_for_map/children_{territory_id}.json')
    # раскрываем json
    res = pd.json_normalize(children['results'], max_level=0)
    res_vill = pd.concat([res.drop('parent', axis='columns'), 
                          pd.json_normalize(res.parent).add_prefix('parent.')],
                          axis=1)
    # меняем тип колонки с геометрией
    for_use = res_vill.copy()
    all_children = gpd.GeoDataFrame(res_vill, 
                         geometry=[shape(d) for d in for_use.pop("geometry")])
    return all_children


def clip_all_children(all_children):
    a = all_children.copy()
    # объединим, чтобы для каждого ГП/СП был один полигон
    buff = a[a.level==5].dissolve(by='parent.name').set_crs(epsg=4326)
    # вручную режем по границе КАЖДОГО ГП/СП, 
    # тк если на втором месте в .clip() - датафрейм, то режет по его общей внешней границе
    gp_sp_orig = a[a.level==4].set_crs(epsg=4326).sort_values(by='name')
    
    np_clipped = pd.DataFrame()
    for i in range(buff.shape[0]):
        clipped = gpd.clip(buff.iloc[i:i+1], 
                             gp_sp_orig.iloc[i:i+1]
                            )
        np_clipped = pd.concat([np_clipped,clipped])
        
    return np_clipped


def get_first_children_data(session, territory_id=34, from_api=False):
    # 34 -- Всеволожский муниципальный район
    if from_api:
        url=f'http://10.32.1.107:5300/api/v1/territory/indicator_values?parent_id={territory_id}&indicator_ids=1&last_only=false'
        r = session.get(url)
        first_children = pd.DataFrame(r.json())
        #first_children.to_json(f'json_api_for_map/first_children_{territory_id}.json')
    else:
        first_children = pd.read_json(f'json_api_for_map/first_children_{territory_id}.json')
    # df: type, geometry, properties(territory_id,name,...)
    vills_with_geom = pd.json_normalize(first_children['features'], max_level=0)
    # df: territory_id, name, indicators
    vills_with_pop = pd.json_normalize(vills_with_geom['properties'], max_level=0)

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
    # вручную режем по границе КАЖДОГО ГП/СП
    np_clipped = clip_all_children(all_children)
    vills_in_area = np_clipped.reset_index().set_crs(epsg=4326)[['parent.name','geometry']]
    fin_vills_full = df.merge(vills_in_area, left_on='name', right_on='parent.name')
    
    return df, fin_vills_full, all_children


def plot_density_data(session, territory_id=34, year=2023, from_api=False):
    # для применения mapbox_style="light" (названия поверх цветных границ)
    token = 'pk.eyJ1IjoibWFyeWtrayIsImEiOiJjbHV3aW5oeTcwY3c0MmptaWU1Z2Z4dGg0In0.FsgSalNNl7L6AvSDU3tr4w'
    common_color_list = get_color_list()
    # данные о ГП/СП, данные об НП
    df, fin_vills_full, _ = get_density_data(session, territory_id, from_api)
    col_dnst = f'{year}_dnst'
    col_b_dnst = f'{year}_dnst_binned'
    # geometry_x -- полигоны ГП/СП, geometry_y -- точки деревень
    fin_vills = fin_vills_full[['name','geometry_y',col_b_dnst]]
    fin_vills.columns = ['name','geometry',col_b_dnst]
    fin_vills = gpd.GeoDataFrame(fin_vills)
    
    # центральная точка для фиксирования карты
    coord = df.dissolve().to_crs(epsg=6933).centroid.to_crs(epsg=4326).values[0]
    center_ll = [coord.y, coord.x]
    
    # сначала сортируем, потом -- в строку
    df = df.sort_values(col_b_dnst)
    # .astype('str') тк иначе жалуется, если отсутствует категория
    df[col_b_dnst] = df[col_b_dnst].astype('str') 
    # берем НП в таком же порядке родительских названий
    fin_vills = fin_vills.set_index('name').loc[df.name].reset_index()
    
    color_list = [common_color_list[i] for i in df[col_b_dnst].unique()]
    
    # в hover_data не убирается инф-ия об индексе; пусть будет тогда territory_id
    df_ti = df.set_index('territory_id')
    # рисуем сами границы ГП/СП
    fig = px.choropleth_mapbox(data_frame = df_ti, geojson=df_ti['geometry'], 
                               locations=df_ti.index, color=col_b_dnst,
                               hover_name=df_ti.name + '<br>(плотность: ' + df_ti[col_dnst].astype('str')+')', 
                               hover_data = {col_dnst:False, col_b_dnst:False}, 
                               labels={col_dnst:'Плотность', 
                                       col_b_dnst: f'<br>Плотность населения<br>(чел/км\N{SUPERSCRIPT TWO})'},
                               # чтобы макс.плотность была первым цветом (желтая)
                               color_discrete_sequence = color_list, opacity=0.5,
                               zoom=8, center = {"lat": center_ll[0], "lon": center_ll[1]})
    
    fig.update_traces(marker_line_width=0.6, marker_opacity=0.4, marker_line_color='darkgray')
    fig.update_layout(mapbox_style="light", mapbox_accesstoken=token,
                     margin={"r":0,"t":0,"l":0,"b":0},
                     autosize=True)

    # рисуем НП: города/деревни/села/...
    fin_vills[col_b_dnst] = fin_vills[col_b_dnst].astype('str') 
    fig_children = px.choropleth_mapbox(fin_vills, geojson=fin_vills['geometry'], color=col_b_dnst,
                                color_discrete_sequence = color_list, 
                                locations=fin_vills.index, zoom=8, 
                                center = {"lat": center_ll[0], "lon": center_ll[1]}, 
                                opacity=1)
    
    fig_children.update_traces(marker_line_width=0.1, marker_opacity=0.6, marker_line_color='lightgray',
                       showlegend=False, hoverinfo='skip', hovertemplate=None)
    fig_children.update_layout(hovermode=False)
    # объединяем
    fig.add_traces(list(fig_children.select_traces()))
    fig.show()
    
    
def density_data_geojson(session, territory_id=34, from_api=False):
    # данные о ГП/СП и НП
    _, fin_vills_full, _ = get_density_data(session, territory_id=territory_id, 
                                            from_api=from_api)

    full_df = fin_vills_full.drop(columns=['parent.name']).rename(columns={'geometry_x':'geometry_areas',
                                                                          'geometry_y':'geometry_villages'}
                                                                 ).set_geometry('geometry_areas'
                                                                               ).set_index('territory_id')
    # меняем, чтобы удалось преобразовать в geojson
    full_df['geometry_villages'] = full_df['geometry_villages'].astype('str')
    
    return full_df
    