# References:
# http://statsmodels.sourceforge.net/devel/examples/notebooks/generated/glm_formula.html
import statsmodels

import utils
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def combined_data(city, year):
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
    return combined;

def runPoissonRegression(city, year):
    print runPoissonRegression.__name__;
    combined = combined_data(city, year)
    formula = get_formula()
    model = smf.glm(formula=formula, data=combined, family=sm.families.Poisson()).fit()
    print (model.summary())
    print (model.predict(pd.DataFrame({'weekday': [3], 'time': [8], 'lat': [12], 'lon': [-14]})))
    print (model.summary2())
    return model;


def get_formula():
    formula = 'count ~ ';
    formula += 'standardize_lat(lat):standardize_lon(lon)'
    formula += '+ standardize_time(time)'
    formula += '+ standardize_weekday(weekday)'
    print formula
    return formula


# Create zscore for day
arr = range(7)
day_zscores = scipy.stats.zscore(arr)
# Create zscore for time
# number of divisons
divs = (24 * 60 / utils.timediv)
time_arr = np.linspace(0, divs, num=divs)
time_zscores = scipy.stats.zscore(time_arr)
# Create zscore for grids
grid_arr = np.linspace(0, utils.gridnum, num=utils.gridnum)
grid_zscores = scipy.stats.zscore(grid_arr)


def standardize_weekday(d):
    return day_zscores[d]


def standardize_time(t):
    return time_zscores[t]


def standardize_lat(l):
    return grid_zscores[l]


def standardize_lon(l):
    return grid_zscores[l]

result_file_prefix = 'test_results_poisson_'

def runResult(city,year):
    print runResult.__name__;
    model = runPoissonRegression(city, year);
    test_data = utils.init_sample(city, year)
    cnt = 0
    for i, row in test_data.iterrows():
        df = pd.DataFrame(row)
        d = {'weekday': [df[i].weekday], 'time': [df[i].time], 'lat': [df[i].lat], 'lon': [df[i].lon]}
        df1 = pd.DataFrame(d)
        res = model.predict(df1);
        test_data.loc[i, "count"] = res[0]
        cnt += 1
        if not cnt % 100: print cnt

    print test_data
    file = result_file_prefix + city + '_' + str(year) + '.csv'
    test_data.to_csv(path_or_buf=file)


def plot(city,year):
    file = result_file_prefix+ city +'_'+str (year) + '.csv'
    d2 = pd.DataFrame.from_csv(file)
    df = d2.groupby(['lat', 'lon', 'weekday']).agg('sum')
    df = df.reset_index()
    df.sort_values(by=['weekday'], inplace=True)
    df.set_index(keys=['weekday'], drop=False, inplace=True)

    for day in range(7):
        weekday = df.loc[df.weekday == day]
        del weekday['time']
        del weekday['weekday']
        weekday.plot.hexbin(x='lat', y='lon', C='count', gridsize=15)
        plt.show()

def calculate_error(city,year):
    actual = combined_data(city,year);
    file = result_file_prefix +city + '_' + str(year) + '.csv'
    test = pd.read_csv(file);
    # calculate R-squared error
    # 1 - (Sum of squares in fit/sum of squares from null model)

    del actual['weekday']
    del actual['lon']
    del actual['lat']

    del test['weekday']
    del test['lon']
    del test['lat']
    l1 = actual.as_matrix(columns=['count'])
    l2 = test.as_matrix(columns=['count'])

    avg = np.average(l1)
    print l2

    ssd1 = 0
    ssd2 = 0
    import math
    for i in range(len(l1)):
        s = math.pow(l1[i]-l2[i],2)
        ssd1 += s
        s = math.pow(l2[i]-avg,2)
        ssd2 +=s

    err = 1 - ssd1/ssd2
    print err
    return;


runResult('nyc',2012)
plot('nyc',2012)
calculate_error('nyc',2012);

