import statsmodels.api as sm
import statsmodels.nonparametric.kde
import statsmodels.nonparametric.kernel_density
import statsmodels.nonparametric.kernel_regression
import pandas as pd
import utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches;
import seaborn as sns;
op_file = 'kde_op.csv'
img_loc = 'Images/kde_results/'

def run_kde(city,year):
    file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '.csv'
    data = pd.read_csv(file)
    del data['d_i']
    del data['death']
    del data['injury']

    arr = data.as_matrix(columns=data.columns[1:])
    ndarr = np.array(arr)

    dens_u = sm.nonparametric.KDEMultivariate(ndarr,var_type = 'ccc', bw = 'normal_reference')
    test = utils.init_sample(city,year);

    del test['count']
    test['p'] = 0

    test_arr = test.as_matrix(columns=data.columns[1:])
    test_arr = np.array(test_arr)
    for i, row in test.iterrows():
        a = row.as_matrix();
        a = a.tolist()
        del a[-1]
        del a[0]
        res = dens_u.pdf(a)
        test.loc[i, "p"] = res
    file =  utils.Data_dir + '/' + city + '_' + str(year) +'_'+op_file;
    test.to_csv(file)

def plot(city,year):
    file = utils.Data_dir + '/' + city + '_' + str(year) +'_'+op_file;
    data = pd.read_csv(file)
    for day in range(7):
        weekday = data.loc[data.weekday == day]
        del weekday['time']
        del weekday['weekday']
        weekday.plot.hexbin(x='lat', y='lon', C='p', gridsize=18)
        #s = 'Probability values ;Day of week:'+str(day)+' , '+city+', year '+str(year)
        #patch = mpatches.Patch(color='blue', label=s)
        #plt.legend(handles=[patch])
        #fig_name = 'day_'+str(day)+'.jpg'
        #plt.savefig(fig_name,bbox_inches='tight')
        #plot_actual_by_day(city,year,day)


def plot_actual_by_day(city,year,day):
    file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '.csv'
    data = pd.read_csv(file)
    data = data[data['weekday']==day]
    sns.jointplot('lat', 'lon', data=data, kind="kde");
    s = 'Density of actual occurences'
    patch = mpatches.Patch(color='blue', label=s)
    plt.legend(handles=[patch])
    #fig_name = img_loc + 'day_' + str(day) + '_actual.jpg'
    #plt.savefig(fig_name, bbox_inches='tight')
    plt.show()
    return;


def plot2(city,year):
    file = utils.Data_dir + '/' + city + '_' + str(year) +'_'+op_file;
    data = pd.read_csv(file)
    del data['lat']
    del data['lon']
    data.plot.hexbin( x='weekday',y='time', C='p', gridsize=13)
    plt.show()

def calculate_errors(city,year):
   all_data = utils.combined_data(city,year)
   file = utils.Data_dir + '/' + city + '_' + str(year) +'_'+op_file;
   kde_op = pd.read_csv(file, index_col=0)
   file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '.csv'
   positives = pd.read_csv(file)
   num_events = len(positives)

   #divide the counts by total events
   for i,row in all_data.iterrows() :
       all_data.loc[i, "count"] = float(all_data.loc[i, "count"])/num_events

   all_data.rename(columns={'count': 'p'}, inplace=True)
   combined = all_data.merge(kde_op,how='left', on=['weekday', 'time', 'lat', 'lon'])

   combined['diff'] = 0
   combined['rel_error'] = 0
   count = 0
   for i,row in combined.iterrows():
       combined.loc[i, "diff"] = abs(combined.loc[i, "p_x"] - combined.loc[i, "p_y"])
       if combined.loc[i, "p_x"]!= 0 :
           combined.loc[i, "rel_error"] = abs(combined.loc[i, "p_x"] - combined.loc[i, "p_y"]/(combined.loc[i, "p_x"])*100)
           count +=1

   del combined['p_x']
   del combined['p_y']

   num_points = len(combined)
   z = combined['diff'].values.sum();
   y = combined['rel_error'].values.sum();

   avg_error = z/num_points
   avg_rel_error = y/count
   max_error = combined['diff'].values.max()
   min_error = combined['diff'].values.min()

   print 'Year'+str(year)
   print 'Average error ' + str(avg_error)
   print 'Average relative error ' + str(avg_rel_error)
   print 'Max error' + str(max_error)
   print 'Min error' + str(min_error)
   return;

for y in utils.valid_years:
    print '======'
    run_kde('nyc',y);
    calculate_errors('nyc',y)
    file = utils.Data_dir + '/' + 'nyc' + '_' + str(y) + '_' + op_file;
    df1 = pd.read_csv(file)
    print df1['p'].values.max()
    print df1['p'].values.min()
    print '======'

