#для работы с таблицами
import numpy as np
import pandas as pd

import os

#для полного копирования данных
import copy

#для создания запросов к api
import requests

#для работы с json-файлами
import json
#для работы с json-файлами. Требование пандаса
from io import StringIO

terr_api = os.environ.get('TERRITORY_API')
file_path = 'app_package/src/for_val_ident/'
#признаки, которые не делятся на население
non_pop_features = ['livarea', 'consnewapt', 'avgsalary', 'avgemployers', 'pollutcapturedperc', 'harvest', 'litstreetperc']

def get_oktmo_level(territory_id: int) -> int:
	"""
	Получает territory_id региона (если рассматриваются МО уровня 4), код ОКТМО и уровень для заданной по territory_id территории

	Возвращает tuple(int, int, int), в котором:
	- элемент 0 - territory_id региона, в котором расположено данное поселение или None для районов;
	- элемент 1 - код ОКТМО для территории;
	- элемент 2 - уровень территории

	:param int territory_id: id территории, чьё ОКТМО необходимо получить
	"""

	URL = terr_api + f"/api/v1/territory/{territory_id}"
	r = requests.get(url = URL)
	data = r.json()
	level = data['level']
	if level == 3:
		oktmo = int(data['oktmo_code'])
	elif level == 4:
		parent_id = data['parent']['id']
		URL2 = terr_api + f"/api/v1/territory/{parent_id}"
		parent_r = requests.get(url = URL2)
		region_id = parent_r.json()['parent']['id']
		return region_id, None, level
	else:
		oktmo = None
	return None, oktmo, level

def loc_counts(loc_data: pd.Series, grid_coeffs: pd.DataFrame) -> pd.DataFrame:
	
	"""
	Считает показатели удовлетворённости жителей определённой локации по различным уровням ценностей, связанных с различными идентичностями

	Возвращает pd.DataFrame по данной территории

	:param pd.Series loc_data: pd.Series значений индикаторов для данной локации
	:param pd.DataFrame grid_coeffs: Таблица с коэффициентами для различных индикаторов каждой клетки
	"""

	n_coeffs = copy.deepcopy(grid_coeffs)
	for col in grid_coeffs.keys():
		for row in grid_coeffs[col].keys():
			cell_sum = 0
			if type(grid_coeffs[col][row]) == dict:
				for indicator in grid_coeffs[col][row].keys():
					cell_sum += loc_data[indicator] * grid_coeffs[col][row][indicator]
				n_coeffs.loc[row, col] = cell_sum
	return pd.DataFrame(n_coeffs).reindex(['dev', 'soc', 'bas'])

def tab_to_ser(df: pd.DataFrame, loc = None) -> pd.Series:
	"""
	Переводит показатели удовлетворённости жителей определённой локации по различным уровням ценностей, связанных с различными
	идентичностями, из формата датафрейма в формат pd.Series, что позволит удобно сравнить значения, полученные внутри одного региона

	Возвращает pd.Series по данной территории (с именем, соответсвующим ОКТМО данной территории, или без имени (в зависимости от значения
	параметра loc))

	:param pd.DataFrame df: датафрейм показателей данной локации
	:param int loc: код октмо данной локации, по умолчанию имеет значение None. Если оно таким и остаётся, возвращаемый pd.Series не имеет
	имени
	"""

	sr = pd.Series(df.values.flatten(), index = ['comm_dev', 'soc_workers_dev', 'soc_old_dev', 'soc_parents_dev', 'loc_dev',
												 'comm_soc', 'soc_workers_soc', 'soc_old_soc', 'soc_parents_soc', 'loc_soc',
												 'comm_bas', 'soc_workers_bas', 'soc_old_bas', 'soc_parents_bas', 'loc_bas'])
	if loc:
		sr.name = loc
	return sr

def reg_df_to_tab(reg_df: pd.DataFrame, grid_coeffs: pd.DataFrame) -> pd.DataFrame:
	"""
	Подсчитывает удовлетворённость жителей всех муниципальных районов определённого региона по различным уровням ценностей, связанных с
	различными идентичностями, исходя из переданных соцэкономпоказателей всех районов данного региона. Показатели распределены по таблице
	"ценностей/идентичностей" и имеют коэффициенты. Это распределение и коэффициенты должны быть переданы в аттрибут grid_coeffs

	Возвращает датафрейм, в котором у каждого муниципального района региона есть своя строка, а в столбцы вписаны значения для всех
	клеток таблицы "ценностей/идентичностей"

	:param pd.DataFrame reg_df: датафрейм со значениями соцэконом индикаторов для каждого района региона
	:param pd.DataFrame grid_coeffs: Таблица с коэффициентами для различных индикаторов каждой клетки
	"""

	muni1 = reg_df.index.min()
	reg_tab = pd.DataFrame(tab_to_ser(loc_counts(reg_df.loc[muni1], grid_coeffs), muni1))
	for loc, row in reg_df.iterrows():
		if loc == muni1:
			continue
		reg_tab[loc] = tab_to_ser(loc_counts(row, grid_coeffs))
	return reg_tab.T

def reg_df_to_tab_updated(reg_df: pd.DataFrame, grid_coeffs: pd.DataFrame) -> pd.DataFrame:
	"""
	Подсчитывает удовлетворённость жителей всех муниципальных районов определённого региона по различным уровням ценностей, связанных с
	различными идентичностями, исходя из переданных соцэкономпоказателей всех районов данного региона. Показатели распределены по таблице
	"ценностей/идентичностей" и имеют коэффициенты. Это распределение и коэффициенты должны быть переданы в аттрибут grid_coeffs

	Возвращает датафрейм, в котором у каждого муниципального района региона есть своя строка, а в столбцы вписаны значения для всех
	клеток таблицы "ценностей/идентичностей"

	:param pd.DataFrame reg_df: датафрейм со значениями соцэконом индикаторов для каждого района региона
	:param pd.DataFrame grid_coeffs: Таблица с коэффициентами для различных индикаторов каждой клетки
	"""

	muni1 = reg_df.index.min()
	reg_tab = pd.DataFrame(data = np.zeros((reg_df.shape[0], 15)), index = reg_df.index)
	print(reg_tab)
	return 0
	reg_tab.apply()
	for loc, row in reg_df.iterrows():
		if loc == muni1:
			continue
		reg_tab[loc] = tab_to_ser(loc_counts(row, grid_coeffs))
	return reg_tab.T

def ser_to_tab(sr: pd.Series, grid_coeffs: pd.DataFrame) -> pd.DataFrame:
	"""
	Переводит показатели удовлетворённости жителей определённой локации по различным уровням ценностей, связанных с различными
	идентичностями, из формата pd.Series в формат датафрейма, который необходим для отображения таблицы

	Возвращает pd.DataFrame по данной территории

	:param pd.Series sr: pd.Series показателей данной локации
	:param pd.DataFrame grid_coeffs: Таблица с коэффициентами для различных индикаторов каждой клетки, нужен для удобства формирования
	таблицы по данной территории из списка её показателей
	"""

	n_coeffs = copy.deepcopy(grid_coeffs)
	for ix in sr.index:
		n_coeffs.loc[ix[-3:], ix[:-4]] = sr.loc[ix]
	return n_coeffs

def recount_data_for_reg(reg_df: pd.DataFrame) -> pd.DataFrame:
	"""
	Растягивает значения каждой клетки каждого района в пределах региона в промежуток от 0 до 1. То есть лучшее значение конкретной клетки
	в регионе становится 1.0, а худшее - 0.0; и так для каждой клетки

	Возвращает нормированный таким образом датафрейм показателей удовлетворённости -//-

	:param pd.DataFrame reg_df: датафрейм изначально полученных значений показателей удовлетворённости -//- для каждого района в регионе
	"""

	return reg_df.apply(lambda s: (s - s.min()) / (s.max() - s.min()), axis = 0)

def change_features(oktmo, changes_dict, reg_df):
	return reg_df

def color_intensity(row):
	if pd.isnull(row['distr_vals']):
		return None
	elif row['distr_vals'] >= row['reg_means']:
		return (row['distr_vals'] - row['reg_means']) / (row['reg_maxs'] - row['reg_means'])
	else:
		return -1 * (row['reg_means'] - row['distr_vals']) / (row['reg_means'] - row['reg_mins'])

#Final function
def muni_tab(territory_id: int, feature_changed = False, changes_dict = "") -> json:
	"""
	Составляет таблицу показателей удовлетворённости жителей определённой локации по различным уровням ценностей, связанных с различными идентичностями
	для заданного района, городского или сельского поселений.

	Возвращает json с таблицей

	:param int territory_id: id территории, по которому будет строиться таблица
	:param bool feature_changed: флажок, указывающий на то, изменил ли пользователь значения каких-то факторов
	:param dict changes_dict: словарь изменений в показатели, внесённых пользователем. На данный момент ожидается следующий формат:
	{"<Название 1ого изменённого индикатора>": <новое значение>,
	 "<Название 2ого изменённого индикатора>": <новое значение>,
	 ...
	 "<Название последнего изменённого индикатора>": <новое значение>}
	"""

	#34 - это Всеволожский район

	#получаем id региона, oktmo и уровень данного муниципального образования
	region_id, oktmo, level = get_oktmo_level(territory_id)
	
	if level == 3:
		##ДЛЯ РАЙОНОВ
		#вычисляем октмо региона
		reg_oktmo = oktmo - (oktmo % 1000000)

		#загружаем общую таблицу
		full_df = pd.read_csv(f'{file_path}full_df4.csv', sep = ';', index_col = 0)

		#получаем таблицу региона
		reg_df = full_df[(full_df.index >= reg_oktmo) & (full_df.index < reg_oktmo + 1000000)]

		#проверка флажка об изменениях от пользователя и пересчёт таблицы (при необходимости)
		if feature_changed:
			changes_dict = json.loads(changes_dict)
			print(changes_dict)
			reg_df = change_features(oktmo, changes_dict, reg_df)

		#здесь надо сформировать таблицу с данными по доп модификаторам для районов данного региона
		#reg_df2 = get_features_from_db(territory_id)

		##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
		#для этого загрузим таблицу коэффициентов
		grid_coeffs = pd.read_json(f'{file_path}grid_coeffs.json').reindex(['dev', 'soc', 'bas'])
		reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

		#нормализуем полученные значения по региону
		reg_tab = recount_data_for_reg(reg_tab)

		#вычислим средние, максимальные и минимальные показатели по региону после нормализации
		reg_means = reg_tab.mean()
		reg_maxs = reg_tab.max()
		reg_mins = reg_tab.min()

		#Переведём вычисленные по клеточкам значения в массивы из самого значения для данного района, среднего значения по региону и интенсивности цвета
		#(положительная интенсивность - для зелёного цвета; отрицательная - для красного)
		distr_ser = reg_tab.loc[oktmo]
		distr_ser = pd.DataFrame({'distr_vals': distr_ser, 'reg_means': reg_means, 'reg_maxs': reg_maxs, 'reg_mins': reg_mins})
		distr_ser['color'] = distr_ser.apply(color_intensity, axis = 1)

		distr_ser['finals'] = distr_ser.apply(lambda x: [x['distr_vals'], x['reg_means'], x['color']], axis = 1)

		#получим обратно значения клеточек для нашего мун. образования
		tab = ser_to_tab(distr_ser['finals'], grid_coeffs)
		
		#теперь выдаём это, как json, и всё
		return tab.to_json()

	elif level == 4:
		##ДЛЯ ГП/СП
		#загружаем таблицу поселений региона
		reg_df = pd.read_csv(f'{file_path}df_{region_id}_{level}.csv', sep = ';', index_col = 0)

		#проверка флажка об изменениях от пользователя и пересчёт таблицы (при необходимости)
		if feature_changed:
			changes_dict = json.loads(changes_dict)
			print(changes_dict)
			reg_df = change_features(territory_id, changes_dict, reg_df)

		#здесь надо сформировать таблицу с данными по доп модификаторам для районов данного региона
		#reg_df2 = get_features_from_db(territory_id)

		##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
		#для этого загрузим таблицу коэффициентов
		grid_coeffs = pd.read_json(f'{file_path}grid_coeffs_4.json').reindex(['dev', 'soc', 'bas'])
		reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

		#нормализуем полученные значения по региону
		reg_tab = recount_data_for_reg(reg_tab)

		#вычислим средние, максимальные и минимальные показатели по региону после нормализации
		reg_means = reg_tab.mean()
		reg_maxs = reg_tab.max()
		reg_mins = reg_tab.min()

		#Переведём вычисленные по клеточкам значения в массивы из самого значения для данного района, среднего значения по региону и интенсивности цвета
		#(положительная интенсивность - для зелёного цвета; отрицательная - для красного)
		distr_ser = reg_tab.loc[territory_id]
		distr_ser = pd.DataFrame({'distr_vals': distr_ser, 'reg_means': reg_means, 'reg_maxs': reg_maxs, 'reg_mins': reg_mins})
		distr_ser['color'] = distr_ser.apply(color_intensity, axis = 1)

		distr_ser['finals'] = distr_ser.apply(lambda x: [x['distr_vals'], x['reg_means'], x['color']], axis = 1)

		#получим обратно значения клеточек для нашего мун. образования
		tab = ser_to_tab(distr_ser['finals'], grid_coeffs)
		
		#теперь выдаём это, как json, и всё
		return tab.to_json()
	else:
		##ДЛЯ ДРУГИХ УРОВНЕЙ
		raise ValueError(f'Localities of this level (given: {level}) are not supported in this request')

#Функция для комплексной выдачи таблицы
def ch_muni_tab(parent_id: int, show_level: int) -> json:
	"""
	Составляет таблицу показателей удовлетворённости жителей для всех локаций определённого уровня по parent_id по различным уровням ценностей, связанных с
	различными идентичностями

	Возвращает json с таблицей

	:param int parent_id: id территории, по по "детям" которой будут строиться таблицы
	:param int show_level: id уровня "детских" территорий, для которых будут строиться таблицы
	"""
	#уровень 3 - районы
	if (show_level != 3) & (show_level != 4):
		raise ValueError(f'Localities of this level (given: {show_level}) are not supported in this request')
	URL = terr_api + f"/api/v1/territory/{parent_id}"
	r = requests.get(url = URL)
	level = r.json()['level']
	if show_level <= level:
		raise ValueError(f'Show level (given: {show_level}) must be > parent territory level (given: parent_id = {parent_id} with level {level})')
	if level == 1:
		raise ValueError(f'Creating matrices for each district or urban/rural settlement in Russia at a time is not supported')

	if (level == 2) & (show_level == 3):
		##Для районов в регионе
		#Получаем oktmo региона
		URL = terr_api + f"/api/v1/territories_without_geometry?parent_id={parent_id}&get_all_levels=false&cities_only=false&ordering=asc&page=1&page_size=1"
		r = requests.get(url = URL)
		oktmo = int(r.json()['results'][0]['oktmo_code'])
		reg_oktmo = oktmo - (oktmo % 1000000)

		#загружаем общую таблицу
		full_df = pd.read_csv(f'{file_path}full_df4.csv', sep = ';', index_col = 0)

		#получаем таблицу региона
		reg_df = full_df[(full_df.index >= reg_oktmo) & (full_df.index < reg_oktmo + 1000000)]

		##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
		#для этого загрузим таблицу коэффициентов
		grid_coeffs = pd.read_json(f'{file_path}grid_coeffs.json').reindex(['dev', 'soc', 'bas'])
		reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

		#нормализуем полученные значения по региону
		reg_tab = recount_data_for_reg(reg_tab)

		return reg_tab.to_json()

	elif (level == 2) & (show_level == 4):
		##Для ГП/СП в регионе
		#загружаем таблицу поселений региона
		reg_df = pd.read_csv(f'{file_path}df_{parent_id}_{show_level}.csv', sep = ';', index_col = 0)

		##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
		#для этого загрузим таблицу коэффициентов
		grid_coeffs = pd.read_json(f'{file_path}grid_coeffs_4.json').reindex(['dev', 'soc', 'bas'])
		reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

		#нормализуем полученные значения по региону
		reg_tab = recount_data_for_reg(reg_tab)

		#теперь выдаём это, как json, и всё
		return reg_tab.to_json()

	elif (level == 3) & (show_level == 4):
		##Для ГП/СП в районе
		#Вычисляем id региона
		region_id = r.json()['parent']['id']
		#Получаем список territory_id ГП/СП региона, находящихся в данном районе
		URL = terr_api + f"/api/v1/all_territories_without_geometry?parent_id={parent_id}&get_all_levels=false&cities_only=false&ordering=asc"
		r = requests.get(url = URL)
		settlements = r.json()
		for i in range(len(settlements)):
			settlements[i] = settlements[i]['territory_id']
		#загружаем таблицу поселений региона
		reg_df = pd.read_csv(f'{file_path}df_{region_id}_{show_level}.csv', sep = ';', index_col = 0)

		##переводим таблицу индикаторов региона в таблицу значений "клеточек" для региона
		#для этого загрузим таблицу коэффициентов
		grid_coeffs = pd.read_json(f'{file_path}grid_coeffs_4.json').reindex(['dev', 'soc', 'bas'])
		reg_tab = reg_df_to_tab(reg_df, grid_coeffs)

		#нормализуем полученные значения по региону
		reg_tab = recount_data_for_reg(reg_tab)

		#оставим только ГП/СП данного района
		reg_tab = reg_tab[reg_tab.index.isin(settlements)]
		reg_tab = reg_tab.sort_index()

		#теперь выдаём это, как json, и всё
		return reg_tab.to_json()
	pass

"""РАЗДЕЛ РЕКОМЕНДАЦИЙ"""

def determine_bad(val_ident_ser):
	#костыль, чтобы работало с null'ами
	to_leave = val_ident_ser.iloc[2:]

	to_leave = to_leave.apply(lambda x: True if x[2] < 0 else False)
	to_leave = to_leave[to_leave].index
	return val_ident_ser[to_leave]

#Функция для рекомендациям по клеточкам
def smart_cell_recommend(val_ident_matrix):
	#Выдаёт 12 главных улучшений факторов, чтобы клеточки стали зеленее

	#Делаем из json'а таблицу
	val_ident_matrix = pd.read_json(StringIO(val_ident_matrix)).reindex(['dev', 'soc', 'bas'])
	#Переводим таблицу в Series
	val_ident_ser = tab_to_ser(val_ident_matrix)
	return determine_bad(val_ident_ser).to_json()

"""подраздел рекомендаций по факторам"""

def feature_recommend(territory_id):
	
