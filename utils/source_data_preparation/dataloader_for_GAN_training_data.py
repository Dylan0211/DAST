"""
这个是最新的版本，负责周中数据到周中数据的context转换的数据生成
"""
from datetime import datetime
from config import all_context_set, daily_separation_point
from models.DAST.time_series_decomposition.RobustSTL import RobustSTL

import pickle
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def normalize(df):
    load_min = df['electricityLoad'].min()
    load_max = df['electricityLoad'].max()
    df['nor_el'] = 0
    df['nor_el'] = (df['electricityLoad'] - load_min) / (load_max - load_min)

    temperature_min = df['temperature'].min()
    temperature_max = df['temperature'].max()
    df['nor_temp'] = 0
    df['nor_temp'] = (df['temperature'] - temperature_min) / (temperature_max - temperature_min)

    return df, load_min, load_max, temperature_min, temperature_max


def build_original_load_table(df):
    begin_time = df.loc[0, 'time']
    end_time = df.loc[df.shape[0] - 1, 'time']
    should_total_hours = (datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S") -
                          datetime.strptime(begin_time, "%Y-%m-%d %H:%M:%S")).days * 24

    time_col = pd.date_range(begin_time, periods=should_total_hours, freq='1h')
    df2 = pd.DataFrame(columns=['time'])
    df2['time'] = time_col
    df['time'] = df['time'].astype('datetime64[ns]')
    df3 = pd.merge(df2, df, how='outer', on=['time'])

    df3.sort_values('time', inplace=True)
    df3.reset_index(drop=True, inplace=True)
    df3.fillna(0, inplace=True)
    return df3[['time', 'electricityLoad', 'temperature']]


def fill_value_into_null(df):
    for index in range(df.shape[0]):
        if df.loc[index, 'temperature'] == 0:
            try:
                df.loc[index, 'temperature'] = df.loc[index - 1, 'temperature']
            except:
                pass

        if df.loc[index, 'electricityLoad'] == 0:
            try:
                df.loc[index, 'electricityLoad'] = df.loc[index - 24, 'electricityLoad']
            except:
                pass
    return df


def context_setting_weather_day(df):
    """
    被pandas map调用
    类似复现 IJCAI：df["hour_level"] = df["hour"].map(time_hour_map)
    x 就直接就是time，不是一整行df

    # 这里用的context是 Ashrae 的Weather day type 24-hour profile plots 提到的8分类：

    winter peak weekday,                1
    winter average weekday,             2
    winter average weekend day/holiday, 3

    summer peak weekday,                4
    summer average weekday,             5
    summer average weekend day/holiday, 6

    spring average weekday,             7
    fall average weekday,               8

    :param x:
    :return:
    """

    #  0. 新添加的几列
    df['8_class'] = -1  # 最后要给每条item填上对应的class
    df['season'] = -1
    df['weekend_flag'] = -1

    # 1. season
    seasons = {
        1: 'Winter',
        2: 'Spring',
        3: 'Summer',
        4: 'Autumn'
    }

    def judge_season(x):
        return (x.month % 12 + 3) // 3

    df['season'] = df['time'].map(judge_season)

    # 2. is weekend？
    def judge_is_weekend_or_weekday(x):
        the_day = x.weekday()
        if the_day > 4:
            return 1  # weekend 是1
        else:
            return 0

    # here to add whether weekend
    df['weekend_flag'] = df['time'].map(judge_is_weekend_or_weekday)  # 0~6 从星期一开始到周日，0是星期一

    # 3. peak
    """
    For example, the summer peak weekday
    can be defined by selecting the five warmest non-holiday
    weekdays during June, July, and August using the actual
    weather raw_data for the calibration period.
    """

    def get_year_column(x):
        return x.year

    top_peak_num = 20  # ashrae 里面的写的， todo 这个按每年的来
    df_temperature = df[['time', 'electricityLoad', 'temperature', 'season', 'weekend_flag']].copy()
    df_temperature['year'] = df_temperature['time'].map(get_year_column)
    for index in range(df_temperature.shape[0]):
        if df_temperature.loc[index, 'temperature'] == 0:  # load 有可能是真关机，但是天气不会==0
            df_temperature.loc[index, 'temperature'] = df_temperature.loc[index - 1, 'temperature']
    df_temperature['hour'] = df_temperature['time'].dt.hour
    df_temperature_12clock = df_temperature[
        df_temperature['hour'] == 12]  # 虽然12点不一定是一天中温度最高的时候，但是某一天12点的温度比其他天高，整体应该也比其他高

    # 3.1 winter peak weekday，这个应该是每年都有, todo 其实每年也有点不合理，但总比全部几年挑几天peak 合理
    the_peak_date_winter_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_winter = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 1].reset_index(
            drop=True)
        top_k_idx = np.array(df_winter['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_winter = df_winter.loc[top_k_idx, 'time']  # 装了气温最高的几天
        for ii in the_peak_date_winter.index:
            the_peak_date_winter_list.append(str(the_peak_date_winter.loc[ii]).split(' ')[0])  # item :'2018-08-01'

    # extract_wanted_days_data(days_list=the_peak_date_winter_list, df=df_temperature)

    # 3.2 summer peak weekday
    the_peak_date_summer_list = []
    for year in pd.unique(df_temperature['year']):
        df_temperature_12clock_this_year = df_temperature_12clock[df_temperature_12clock['year'] == year]
        df_summer = df_temperature_12clock_this_year[df_temperature_12clock_this_year['season'] == 3].reset_index(
            drop=True)
        top_k_idx = np.array(df_summer['temperature']).argsort()[::-1][0: top_peak_num]
        the_peak_date_summer = df_summer.loc[top_k_idx, 'time']  # 装了气温最高的几天
        for ii in the_peak_date_summer.index:
            the_peak_date_summer_list.append(str(the_peak_date_summer.loc[ii]).split(' ')[0])  # item :'2018-08-01'

    # 4. for 循环赋值
    # print('build contextual column for df...')
    for index in range(df.shape[0]):
        # winter相关
        if df.loc[index, 'season'] == 1:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 3
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 2

                # 这个也是要weekday
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_winter_list:
                    df.loc[index, '8_class'] = 1  # 这个放在前两个if后面，因为会有overwrite

        # summer 相关
        elif df.loc[index, 'season'] == 3:
            if df.loc[index, 'weekend_flag'] == 1:
                df.loc[index, '8_class'] = 6
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 5

                # 这个也是要weekday
                if str(df.loc[index, 'time']).split(' ')[0] in the_peak_date_summer_list:
                    df.loc[index, '8_class'] = 4  # 这个放在前两个if后面，因为会有overwrite

        # 春秋
        elif df.loc[index, 'season'] == 2:  # 春
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 7
            else:
                df.loc[index, '8_class'] = 9
        elif df.loc[index, 'season'] == 4:  # 秋
            if df.loc[index, 'weekend_flag'] == 0:
                df.loc[index, '8_class'] = 8
            else:
                df.loc[index, '8_class'] = 10

        else:
            raise ValueError(df.loc[index, 'time'] + '找不到`8分类`')

    return df


def one_hot(data, feature):
    onehot = pd.get_dummies(data[feature])
    df = data.drop(feature, axis=1)
    data = df.join(onehot)
    return data


def create_X_Y(df, seq_length):
    X = []
    Y = []
    for i in range(df.shape[0] - 2 * seq_length):
        X.append(np.array(df.iloc[i: i + seq_length, 6:]))
        Y.append(np.array(df.loc[i + seq_length: i + 2 * seq_length - 1, 'nor_el']))
    X = np.array(X)
    Y = np.array(Y)

    return X, Y


def get_mapped_data_for_GAN_training(df, context_data_dict, staring_timestamp):
    load_mapping_dict = {}
    for context_a in all_context_set:
        for context_b in all_context_set:
            if context_a == context_b:
                continue

            df_context_a = context_data_dict[context_a]['df_context']
            df_context_b = context_data_dict[context_b]['df_context']

            context_a_each_day_data = [[] for i in range(7)]
            context_b_each_day_data = [[] for i in range(7)]
            for i in range(df_context_a.shape[0]):
                if str(df_context_a.loc[i, 'time']).endswith(staring_timestamp):
                    context_a_each_day_data[df_context_a.loc[i, 'weekday_feature']].append(df_context_a.loc[i, 'time'])
            for i in range(df_context_b.shape[0]):
                if str(df_context_b.loc[i, 'time']).endswith(staring_timestamp):
                    context_b_each_day_data[df_context_b.loc[i, 'weekday_feature']].append(df_context_b.loc[i, 'time'])

            # get each time raw_data and mappings
            source_context_data = [[] for i in range(7)]
            for i in range(len(context_a_each_day_data)):  # every weekday raw_data
                for j in range(len(context_a_each_day_data[i])):
                    this_index = df[df['time'] == context_a_each_day_data[i][j]].index[0]
                    this_num = np.array(df.loc[this_index: this_index + 23, 'nor_el_season'])
                    if this_num.shape[0] < 24:
                        continue
                    source_context_data[i].append(this_num)
            target_context_data = [[] for i in range(7)]
            for i in range(len(context_b_each_day_data)):
                for j in range(len(context_b_each_day_data[i])):
                    this_index = df[df['time'] == context_b_each_day_data[i][j]].index[0]
                    this_num = np.array(df.loc[this_index: this_index + 23, 'nor_el_season'])
                    if this_num.shape[0] < 24:
                        continue
                    target_context_data[i].append(this_num)

            source_load = []
            target_load = []
            for i in range(7):
                for j in range(len(source_context_data[i])):
                    for k in range(len(target_context_data[i])):
                        source_load.append(source_context_data[i][j])
                        target_load.append(target_context_data[i][k])

            source_load = np.array(source_load)
            target_load = np.array(target_load)

            load_mapping_dict.update({(context_a, context_b): (source_load, target_load)})

    return load_mapping_dict


def get_each_day_load(df, staring_timestamp):
    context_data_dict = {}
    for context in all_context_set:
        # note: get raw_data in one or multiple contexts as source raw_data
        df_context = df[df['season'] == context]
        df_context.reset_index(drop=True, inplace=True)

        each_day_start_index_list = []
        for i in range(df_context.shape[0]):
            if str(df_context.loc[i, 'time']).endswith(staring_timestamp):
                each_day_start_index_list.append(i)
        context_data = []
        normalized_context_data = []
        for i in range(len(each_day_start_index_list)):
            tmp_data = df_context.loc[each_day_start_index_list[i]: each_day_start_index_list[i] + 23, 'nor_el_season']

            normalized_tmp_data = df_context.loc[each_day_start_index_list[i]: each_day_start_index_list[i] + 23, 'electricityLoad']
            max_load = normalized_tmp_data.max()
            min_load = normalized_tmp_data.min()
            if max_load != min_load:
                normalized_tmp_data = [(load - min_load) / (max_load - min_load) for load in normalized_tmp_data]
            else:
                normalized_tmp_data = np.zeros_like(normalized_tmp_data)

            if tmp_data.shape[0] < 24:
                continue
            if df_context.loc[each_day_start_index_list[i] + 23, 'season'].item() != context:
                continue
            context_data.append(tmp_data)
            normalized_context_data.append(normalized_tmp_data)
        context_data = np.array(context_data)
        normalized_context_data = np.array(normalized_context_data)

        # note: test
        # for i in range(context_data.shape[0]):
        #     load_trace = context_data[i]
        #     plt.plot(range(load_trace.shape[0]), load_trace)
        #     plt.show()

        start_index = each_day_start_index_list[0]
        end_index = each_day_start_index_list[context_data.shape[0] - 1]
        df_context = df_context.loc[start_index: end_index + 23, :]
        df_context.reset_index(drop=True, inplace=True)

        context_data_dict.update({context: {
            'context_data': context_data,
            'normalized_context_data': normalized_context_data,
            'df_context': df_context,
        }})

    return context_data_dict


def read_data(target_building_name):
    # load raw_data
    data = pd.read_csv(r'./raw_data/meters/cleaned/electricity_cleaned.csv')
    df = data.loc[:, ['timestamp', target_building_name]]
    df.columns = ['time', 'electricityLoad']

    # preprocess data
    building_category = target_building_name[:target_building_name.index('_')]
    temp = pd.read_csv(r'./raw_data/weather/weather.csv')
    temp = temp[temp['site_id'] == building_category]
    temp = temp.loc[:, ['timestamp', 'airTemperature']]
    temp.reset_index(drop=True, inplace=True)
    temp.columns = ['time', 'temperature']
    df = pd.merge(df, temp)
    df.sort_values('time', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # fill in missing val
    df = build_original_load_table(df=df)
    df = fill_value_into_null(df=df)

    # config context
    df = context_setting_weather_day(df=df)

    # normalize
    df, _, _, _, _ = normalize(df=df)

    # decomposition load traces
    nor_el = df.loc[:, 'nor_el']
    result = RobustSTL(nor_el, 24, reg1=10.0, reg2=0.5, K=0, H=1, dn1=1., dn2=1., ds1=50., ds2=1.)
    if np.max(result[1] + result[2]) > 1 or np.min(result[1] + result[2]) < 0:
        df['nor_el_season'] = nor_el
        df['nor_el_remainder'] = np.zeros_like(nor_el)
    else:
        df['nor_el_season'] = result[1] + result[2]
        df['nor_el_remainder'] = result[3]

    # get time feature and weekday feature
    time_feature = []
    weekday_feature = []
    for i in range(df.shape[0]):
        time_feature_i = str(df.loc[i, :]['time']).split(' ')[-1]
        weekday_feature_i = time.strptime(str(df.loc[i, :]['time']), "%Y-%m-%d %H:%M:%S").tm_wday
        time_feature.append(time_feature_i)
        weekday_feature.append(weekday_feature_i)
    df['time_feature'] = np.array(time_feature)
    df['weekday_feature'] = np.array(weekday_feature)

    starting_timestamp = '{:02}:00:00'.format(daily_separation_point)
    index_list = [idx for idx in range(df.shape[0]) if str(df.iloc[idx]['time']).endswith(starting_timestamp)]
    df = df.iloc[index_list[0]: index_list[-1]]
    df.reset_index(drop=True, inplace=True)

    print('starting timestamp: {}'.format(starting_timestamp))

    season_avg_temp_dict = {}
    for season_id in [1, 2, 3, 4]:
        this_season_avg_temp = df[df['season'] == season_id].loc[:, 'temperature'].mean()
        season_avg_temp_dict.update({season_id: this_season_avg_temp})

    context_data_dict = get_each_day_load(df=df, staring_timestamp=starting_timestamp)

    # note: get raw_data in source and target context as GAN training raw_data
    load_mapping_dict = get_mapped_data_for_GAN_training(df=df, context_data_dict=context_data_dict,
                                                         staring_timestamp=starting_timestamp)

    # save raw_data
    save_dict = {
        'df': df,
        'context_data_dict': context_data_dict,
        'load_mapping_dict': load_mapping_dict,
        'season_avg_temp_dict': season_avg_temp_dict,
    }

    with open(r'./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(target_building_name), 'wb') as w:
        pickle.dump(save_dict, w)
    print('    raw_data saved at: ./tmp_data/building_tmp_data/{}_GAN_training_data_dict.pkl'.format(target_building_name))


if __name__ == '__main__':
    read_data(target_building_name='Fox_assembly_Boyce')
