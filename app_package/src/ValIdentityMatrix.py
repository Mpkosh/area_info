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

terr_api = os.environ.get('TERRITORY_API')
file_path = 'app_package/src/for_val_ident/'
#признаки, которые не делятся на население
non_pop_features = ['livarea', 'consnewapt', 'avgsalary', 'avgemployers', 'pollutcapturedperc', 'harvest', 'litstreetperc']

def get_oktmo(territory_id: int) -> int:
	"""
	Получает код ОКТМО для заданной по territory_id территории

	Возвращает int, соответствующий коду ОКТМО для территории

	:param int territory_id: id территории, чьё ОКТМО необходимо получить
	"""

	URL = terr_api + f"/api/v1/territory/{territory_id}"
	r = requests.get(url = URL)
	return int(r.json()['oktmo_code'])

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
	для заданного района

	Возвращает json с таблицей

	:param int territory_id: id территории (района), по которому будет строиться таблица
	:param bool feature_changed: флажок, указывающий на то, изменил ли пользователь значения каких-то факторов
	:param dict changes_dict: словарь изменений в показатели, внесённых пользователем. На данный момент ожидается следующий формат:
	{"<Название 1ого изменённого индикатора>": <новое значение>,
	 "<Название 2ого изменённого индикатора>": <новое значение>,
	 ...
	 "<Название последнего изменённого индикатора>": <новое значение>}
	"""

	#34 - это Всеволожский район

	#получаем oktmo данного муниципального образования
	oktmo = get_oktmo(territory_id)
	#получаем oktmo региона, в котором данное мун. образование находится
	reg_oktmo = oktmo - (oktmo % 1000000)

	if reg_oktmo != 41000000 or True:

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

	else:
		pass
