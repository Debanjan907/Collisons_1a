# Read data from CSV file
import csv
import numpy as np;
import utils;

##############################
# set filename
Filename = utils.Data_dir+'NYPD_Motor_Vehicle_Collisions_1.csv';
# set city
City = 'nyc'
gridnum = utils.gridnum;  #
timediv = utils.timediv  # 30 mins

##############################
# Input File format :
# 0 Date
# 1 Time
# 4 Latitude
# 5 Longitude
# 10 Total Injury
# 11 Total Killed
# 12 Ped Injured
# 13 Ped Killed
# 14 Cyc Injured
# 15 Cyc Killed
# 16 Motorist Injured
# 17 Motorist Killed
##############################
# Output file format

# File : city_year.csv
# Date ( mm/dd/yyyy)
# Time
# Lat
# Longitude
# Death or Injury = ( 0 or 1 )
# Number of injuries
# Number of deaths


##############################
lat_lon_box = dict();
lat_lon_box['nyc'] = [40.917577, -74.25909, 40.477399, -73.70000]
data_by_year = dict()

class lat_long_validator:
    city = '';
    min_lat = 0;
    max_lat = 0;
    min_lon = 0;
    max_lat = 0;

    def __init__(self, city):
        self.city = city
        self.set_validation_bounds();

    def set_validation_bounds(self):
        city = self.city;
        if lat_lon_box[city][2] < lat_lon_box[city][0]:
            self.min_lat = lat_lon_box[city][2]
            self.max_lat = lat_lon_box[city][0]
        else:
            self.min_lat = lat_lon_box[city][0]
            self.max_lat = lat_lon_box[city][2]

        if lat_lon_box[city][1] < lat_lon_box[city][3]:
            self.min_lon = lat_lon_box[city][1]
            self.max_lon = lat_lon_box[city][3]
        else:
            self.min_lon = lat_lon_box[city][3]
            self.max_lon = lat_lon_box[city][1]
        print ("Min Lat" + str(self.min_lat) + " Max Lat:" + str(self.max_lat) + " Min Lon" + str(
            self.min_lon) + " Max Lon" + str(self.max_lon))

    def validate(self, lat, lon):
        # check if empty
        if not lat or not lon:
            return False;

        # convert from String to Float
        lat = float(lat)
        lon = float(lon)

        if lat < self.min_lat or lat > self.max_lat:
            return False;
        if lon < self.min_lon or lon > self.max_lon:
            return False;
        return True;


def get_year(date):
    # format "mm/dd/yyyy"
    parts = date.split('/')
    return int(parts[2]);


def readData():
    loc_validator = lat_long_validator('nyc')
    with open(Filename, 'r') as csvfile:
        rd = csv.reader(csvfile)
        count = 0;
        row_count = 0;
        for row in rd:
            count += 1;
            # omit the 1st row
            if count == 1:
                continue;

            year = get_year(row[0]);
            time = row[1]
            lat = row[4]
            lon = row[5]

            # check if empty
            if not lat or not lon:
                continue;
            lat = float(row[4])
            lon = float(row[5])

            if not loc_validator.validate(lat, lon):
                continue;

            death = int(row[10])
            injury = int(row[11])
            if (death > 0 or injury > 0):
                death_or_injury = 1
            else:
                death_or_injury = 0;

            print (row[0] + " " + time + "  Lat :" + str(lat) + " Long:" + str(lon) + "  Death/Injury : " + str(
                death_or_injury))
            row_count += 1
            current = []
            current.append(row[0])
            current.append(time)
            current.append(lat)
            current.append(lon)
            current.append(death_or_injury)
            current.append(injury)
            current.append(death)
            if year not in data_by_year or (data_by_year[year] is None):
                data_by_year[year] = []

            data_by_year[year].append(current)

    print(row_count)
    for year in data_by_year.keys():
        year_list = data_by_year[year]
        op_file = utils.Data_dir + '/' + City + '_' + str(year) + '.csv'
        f = open(op_file, 'w')
        try:
            for row in year_list:
                content = [];
                for z in row:
                    content.append(str(z))
                f.write(",".join(content) + "\n")
        finally:
            f.close()


# Create time buckets
# Create location grids

def discretize_loc_time(city='nyc'):
    columns = ['weekday','time','lat','lon','d_i','death','injury']

    valid_years = [2012,2013,2014,2015,2016]
    for year in valid_years:
        inp_filename = utils.Data_dir + '/' + city + '_' + str(year) + '.csv';
        with open(inp_filename, 'r') as csvfile:
            rd = csv.reader(csvfile)
            bounds = utils.get_city_bounds('nyc')
            min_lat = bounds[0]
            max_lat = bounds[2]
            min_lon = bounds[1]
            max_lon = bounds[3]
            lat_step = (max_lat - min_lat) / gridnum
            lon_step = (max_lon - min_lon) / gridnum
            op_file = utils.Data_dir + '/' +city + '_' + str(year) + '_'+ utils.discreet_str + '.csv'
            f = open(op_file, 'w')
            f.write(",".join(columns) + "\n")
            for row in rd:
                content = []
                content.append(utils.get_day_of_week(row[0])) # weekday
                time_in_mins = utils.get_mins(row[1])
                content.append(time_in_mins / timediv);  # time_bucket
                lat = int((float(row[2])-min_lat) / lat_step)
                lon = int((float(row[3])-min_lon) / lon_step)
                content.append(lat)
                content.append(lon)
                content.append(row[4])
                content.append(row[5])
                content.append(row[6])
                str_content =[]
                for c in content:
                    str_content.append(str(c));
                f.write(",".join(str_content) + "\n")
            f.close()

        #create negative samples
        op_file = utils.Data_dir + '/' + city + '_' + str(year) + '_' + utils.discreet_str + '_neg.csv'
        f = open(op_file, 'w')
        f.write(",".join(columns) + "\n")

        for day in range(7):
            for time in range(24 * 60 / utils.timediv):
                for lat in range(utils.gridnum):
                    for lon in range(utils.gridnum):
                        data = [day,time,lat,lon,0,0,0]
                        str_data = []
                        for d in data:
                            str_data.append(str(d));
                        f.write(",".join(str_data) + "\n")
        f.close()
    return;

discretize_loc_time('nyc')
