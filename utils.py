import datetime

def get_mins(time_str) :
    parts = time_str.split(':')
    return (int(parts[0])*60)+(int(parts[1]));

def get_day_of_week(date_str) :
    #date is in form dd/mm/yyyy
    parts = date_str.split('/')
    d = datetime.date(int(parts[2]), int(parts[0]), int(parts[1]))
    return d.weekday()

def get_city_bounds(city) :
    lat_lon_box = dict();
    #lower left corner - upper right corner
    lat_lon_box['nyc'] = [ 40.477399, -74.25909, 40.917577,-73.70000]
    return lat_lon_box[city];

discreet_str ='discreet_loc_time';
Data_dir = 'Data'
gridnum = 10;  #
timediv = 60  # 30 mins
valid_years = [ 2012,2013,2014,2015,2016]
