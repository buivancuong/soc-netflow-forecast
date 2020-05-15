import datetime
import maya


def sensor2index(sensor):
    if sensor == 'sensor-cmc-corp':
        return 'sensor-cmc-corp-'
    elif sensor == 'Sensor-vnpost-01':
        return 'sensor-vnpost-'
    else:
        sensor = sensor.split('-')
        return sensor[0] + '-' + sensor[1] + '-'


def create_es_index(sensor, unix_timestamp):
    unix_timestamp -= (3600*7)      # GMT+7 to GMT0
    prefix = sensor2index(sensor=sensor)
    date = datetime.datetime.fromtimestamp(int(unix_timestamp))
    year = str(date.year)
    month = date.month
    if(month < 10):
        month = '0' + str(month)
    else:
        month = str(month)

    day = date.day
    if(day < 10):
        day = '0' + str(day)
    else:
        day = str(day)

    return str(prefix + year + '.' + month + '.' + day)


def get_true_timestamp(time_stamp):
    time_stamp = maya.parse(time_stamp).datetime()
    time_stamp += datetime.timedelta(hours=7)
    return str(time_stamp)
