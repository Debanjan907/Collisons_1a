###########
# Bounding box of New york City :
#  40.917577, -74.25909,
#  40.477399, -73.70000
############
# Radius of earth 6371.009 Km
# Units used for geographical distance : Km
############
# TODO: Use bigfloat

import utils
import scipy.stats
import csv
import math
import numpy as np;
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt;
import matplotlib.patches as mpatches
from sklearn.neighbors.kde import KernelDensity

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

##################################
# General utility functions :Start
##################################

valid_years = [2012, 2013, 2014, 2015, 2016]


##################################

class location:
    lat = 0;
    lon = 0;
    R = 6371.009

    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def degrees_to_radians_aux(self, value):
        return (value * math.pi) / 180;

    # return distance between the location,and another location point(object)
    def distance(self, loc2):
        dlon = abs(loc2.lon - self.lon)
        dlat = abs(loc2.lat - self.lat)
        mean_lat = (self.lat + loc2.lat) / 2;
        dist = self.R * math.sqrt(self.degrees_to_radians_aux(dlat) ** 2 +
                                  (math.acos(self.degrees_to_radians_aux(mean_lat)) * self.degrees_to_radians_aux(
                                      dlon)) ** 2)
        return dist;


def plot_basic_map(city):
    # 40.917577, -74.25909,
    # 40.477399, -73.70000
    my_map = Basemap(projection='merc',
                     resolution='h',
                     lat_0=40.5,
                     lon_0=-73.7,
                     llcrnrlat=40.0,
                     llcrnrlon=-74.4,
                     urcrnrlat=41.0,
                     urcrnrlon=-73.0)

    my_map.fillcontinents(color='yellow')
    my_map.drawmapboundary(fill_color='aqua')
    my_map.drawcoastlines()
    my_map.drawcountries()
    return my_map;


def plot_points(city, year, map_obj):
    Filename = city + '_' + str(year) + '.csv'
    x = []
    y = []
    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            x.append(float(row[2]))  # latitude
            y.append(float(row[3]))  # longitude

    for i in range(len(x)):
        lon = y[i]
        lat = x[i]
        x1, y1 = map_obj(lon, lat)
        map_obj.plot(x1, y1, marker='o', markersize=1, color='m')


# map_obj = plot_basic_map('nyc')
# plot_points('nyc', 2014, map_obj)
# plt.show()

def create_time_buckets(granularity):
    b = [];
    for i in range((24 * 60) / granularity):
        b.append(i);
    return b;


def create_time_labels(time_ticks, granularity):
    labels = [];
    label_interval = float(24 * 60 / granularity) / (len(time_ticks) - 1)
    z = 0;
    for i in range(len(time_ticks)):
        labels.append(str(z / 60) + ':' + str(z % 60))
        z += int(label_interval * granularity)
    return labels;


def collisons_by_time(city, year):
    Filename = city + '_' + str(year) + '.csv';
    granularity = 30
    x = create_time_buckets(granularity);
    coll = [];
    death_injury = [];

    for i in range(len(x)):
        coll.append(0);
        death_injury.append(0);

    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            time = row[1]
            time_parts = time.split(':');
            hr = int(time_parts[0])
            min = int(time_parts[1])
            index = (hr * 60 + min) / granularity;
            coll[index] += 1
            c = int(row[4])
            if c > 0:
                death_injury[index] += 1;

    ax = plt.axes()
    plt.xlim(0, len(coll))

    # location of x asis labels
    time_ticks = ax.xaxis.get_majorticklocs()
    labels = create_time_labels(time_ticks, granularity)
    ax.xaxis.set_ticklabels(labels)
    plt.plot(x, coll, 'ro-')
    plt.plot(x, death_injury, 'g-')
    red_patch = mpatches.Patch(color='red', label='Collision count')
    green_patch = mpatches.Patch(color='green', label='Death or injury ')
    city_patch = mpatches.Patch(color='black', label=city + ' ' + str(year))
    gran_patch = mpatches.Patch(color='black', label='Time Step :' + str(granularity))
    plt.legend(handles=[city_patch, gran_patch, red_patch, green_patch])
    plt.show()

    plt.plot(x, death_injury, 'gs-')
    plt.legend(handles=[city_patch, gran_patch, green_patch])
    plt.show()
    return;


def collisions_by_time_loc(city, year):
    Filename = city + '_' + str(year) + '.csv';
    granularity = 30
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    time_buckets = create_time_buckets(granularity);
    x = []  # latitude
    y = []  # longitude
    z = []  # time

    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            x.append(float(row[2]))
            y.append(float(row[3]))
            time = row[1]
            time_parts = time.split(':');
            hr = int(time_parts[0])
            min = int(time_parts[1])
            index = (hr * 60 + min) / granularity;
            z.append(index)

    time_ticks = ax.xaxis.get_majorticklocs()
    labels = create_time_labels(time_ticks, granularity)
    ax.set_zlabel(labels)
    ax.scatter(x, y, z, c='r', marker='o')
    red_patch = mpatches.Patch(color='red', label='Collision Instance')
    city_patch = mpatches.Patch(color='black', label=city + ' ' + str(year))
    gran_patch = mpatches.Patch(color='black', label='Time Step :' + str(granularity))
    ax.legend(handles=[city_patch, gran_patch, red_patch])
    plt.show()


'''
for y in years:
    collisons_by_time('nyc', y)
    collisions_by_time_loc('nyc',y);

'''


def week_time_buckets(granularity):
    length = ((24 * 60) / granularity) * 7;
    bucket = np.zeros(length)
    return bucket;


def coll_by_time_and_weekday(city, year):
    Filename = city + '_' + str(year) + '.csv';
    granularity = 30

    time_bucket_length = (24 * 60) / granularity;
    bucket = np.ndarray(shape=[7, time_bucket_length], dtype=int)

    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            weekday = utils.get_day_of_week(row[0])
            time = row[1]
            time_parts = time.split(':');
            hr = int(time_parts[0])
            min = int(time_parts[1])
            index = (hr * 60 + min) / granularity;
            bucket[weekday][index] += 1

    print bucket
    return;


def coll_by_loc_weekday(city, year):
    Filename = city + '_' + str(year) + '.csv';
    # bucket the locations
    bounds = utils.get_city_bounds(city);
    min_lat = bounds[0]
    min_lon = bounds[1]
    max_lat = bounds[2]
    max_lon = bounds[3]

    lat_diff = abs(max_lat - min_lat)
    lon_diff = abs(max_lon - min_lon)
    num_buckets = 100;  # number of buckets for latitude and longitude
    lat_step = lat_diff / num_buckets
    lon_step = lon_diff / num_buckets

    buckets = np.ndarray(shape=[7, 2], dtype=list)
    for i in range(7):
        buckets[i][0] = []
        buckets[i][1] = []
    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)

        for row in rd:
            weekday = utils.get_day_of_week(row[0])
            lat = float(row[2])
            lon = float(row[3])
            lat_pos = float(int((lat - min_lat) / lat_step) * lat_step) + min_lat
            lon_pos = float(int((lon - min_lon) / lon_step) * lon_step) + min_lon
            buckets[weekday][0].append(lat_pos)
            buckets[weekday][1].append(lon_pos)

        print buckets;
        for day in range(7):
            l1 = len(buckets[day][1])
            xy = np.ndarray(shape=[2, l1], dtype=float)
            xy[0] = buckets[day][0]
            xy[1] = buckets[day][1]
            xy = np.vstack([xy[0], xy[1]])
            z = scipy.stats.gaussian_kde(xy)(xy)
            plt.scatter(xy[1], xy[0], c=z, s=100, edgecolor='')
            plt.show()

'''
# coll_by_time_and_weekday('nyc',2014)
# coll_by_loc_weekday('nyc',2015)

def train_kde(city, year):
    Filename = city + '_' + str(year) + '.csv';
    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        d = []
        count = 0
        for row in rd:
            weekday = utils.get_day_of_week(row[0])
            d.append(weekday)
            # time = get_mins(row[1])
            # d.append(time)
            d.append(float(row[2]))
            d.append(float(row[3]))
            count += 1
        data = np.reshape(d, newshape=[count, 3])

    print data;
    kde = KernelDensity(kernel='gaussian').fit(data)
    return kde


# kde = train_kde('nyc',2015);

def run_kde(city, year, kde):
    bounds = utils.get_city_bounds(city)
    min_lat = bounds[0]
    max_lat = bounds[2]
    min_lon = bounds[1]
    max_lon = bounds[3]

    lat, lat_step = np.linspace(min_lat, max_lat, 1000, retstep=True)
    lon, lon_step = np.linspace(min_lon, max_lon, 1000, retstep=True)
    samples = np.ndarray(shape=[7, 1000, 1000])
    for day in range(0, 6, 1):
        for i in range(1000):
            l1 = min_lat + lat_step * i;
            for j in range(1000):
                l2 = min_lon + lon_step * j
                data = [day, i, j]
                res = kde.score_samples(data)
                print 'Result ' + str(res);
                samples[day][i][j] = res;
    print samples;
    # res = kde.score_samples(data)
    # print res
# run_kde('nyc',2014,kde)
'''


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
    print (model.predict(pd.DataFrame( { 'weekday' : [3], 'time':[8], 'lat':[40.5], 'lon':[-74.2]})))
    return;

PoissonRegression('nyc', 2013)
