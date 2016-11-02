# References:
# http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/glm_formula.html

import utils
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np

def runPoissonRegression(city, year):
    file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '.csv'
    data = pd.read_csv(file)
    data['count'] = 1
    data_with_count = data.groupby(['weekday', 'time', 'lat', 'lon']).count()
    data_with_count = data_with_count.reset_index()
    del data_with_count['d_i'];
    del data_with_count['death'];
    del data_with_count['injury']

    file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '_neg.csv'
    all_neg_samples = pd.read_csv(file)
    all_neg_samples['count'] = 0;
    del all_neg_samples['d_i'];
    del all_neg_samples['death'];
    del all_neg_samples['injury']

    combined = all_neg_samples.merge(data_with_count, how='left', on=['weekday', 'time', 'lat', 'lon']).fillna(0)
    del combined['count_x']
    combined.rename(columns={'count_y': 'count'}, inplace=True)
    formula = get_formula()
    model = smf.glm(formula=formula, data=combined, family=sm.families.Poisson()).fit()
    print (model.summary())
    print (model.predict(pd.DataFrame({'weekday': [3], 'time': [8], 'lat': [40.5], 'lon': [-74.2]})))
    return model;


def get_formula():
    weekday = 'div_by_7(weekday)'
    time = 'standardize_time(time)'
    lat = 'standardize_lat(lat)'
    lon = 'standardize_lon(lon)'

    formula = 'count ~ ';
    formula += 'standardize_lat(lat):standardize_lon(lon):standardize_time(time)'
    formula += '+ standardize_time(time)'
    formula += '+ standardize_weekday(weekday) '
    print formula
    return formula


def standardize_weekday(d):
    return d / 7;


def standardize_time(t):
    divs = (24 * 60 / utils.timediv)
    return t / divs;


def standardize_lat(l):
    return abs(l - utils.gridnum/2)/utils.gridnum;

def standardize_lon(l):
    return (l - utils.gridnum/2)/utils.gridnum;

def init_sample(city, year):
    city = 'nyc'
    file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '_neg.csv'
    test = pd.read_csv(file)
    test['count'] = 0
    del test['d_i']
    del test['death']
    del test['injury']
    return test

def runResult():
    for y in utils.valid_years:
        city = 'nyc'
        model = runPoissonRegression(city, y);
        test_data = init_sample(city,y)
        for i,row in test_data.iterrows():
            df = pd.DataFrame(row)
            d = {'weekday': [df[i].weekday], 'time':  [df[i].time], 'lat':  [df[i].lat], 'lon':  [df[i].lon] }
            df1 = pd.DataFrame(d)
            res = model.predict(df1);
            print res[0]
            test_data.loc[i,"count"] = res[0]
            file = '2.csv'
            test_data.to_csv(path_or_buf=file)


def plot():
    file = '2.csv'
    d2 = pd.DataFrame.from_csv(file)
    df = d2.groupby(['lat', 'lon','weekday']).agg('sum')
    df = df.reset_index()
    df.sort_values(by=['weekday'], inplace=True)
    df.set_index(keys=['weekday'], drop=False, inplace=True)

    for day in range(7):
        weekday= df.loc[df.weekday == day]
        del weekday['time']
        del weekday['weekday']
        weekday.plot.hexbin(x='lat', y='lon', C='count')
        plt.show()


runResult()
plot()
