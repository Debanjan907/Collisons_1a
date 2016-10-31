import utils
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def PoissonRegression(city, year):
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
    formula = 'count ~ weekday + time + lat + lon';
    model = smf.glm(formula = formula, data = combined, family = sm.families.Poisson()).fit()
    print (model.summary())
    #print (model.predict(pd.DataFrame( { 'weekday' : [3], 'time':[8], 'lat':[40.5], 'lon':[-74.2]})))
    return model;

for y in utils.valid_years:
    PoissonRegression('nyc',y);
