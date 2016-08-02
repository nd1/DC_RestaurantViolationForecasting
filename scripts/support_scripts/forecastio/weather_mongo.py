'''
Pull daily historical weather data from forecast.io from 5/1/2013 through present day. Save the data to mongodb.

Future development-- stop error on the last date.

20160627 Nicole Donnelly
'''
import json
import datetime as dt
import forecastio
import pytz
import pymongo

api_key= 'add api key'
lat = 38.9072
lng = 77.0369
est = pytz.timezone('US/Eastern')

conn=pymongo.MongoClient()
db = conn.forecastio
records = db.records

def insert(record_date, fio_record):
    weather_data = {}
    weather_data['date'] = str(record_date.date())
    weather_data['maxTemp'] = fio_record.data[0].d['temperatureMax']
    weather_data['minTemp'] = fio_record.data[0].d['temperatureMin']
    if 'dewPoint' in fio_record.data[0].d.keys():
        weather_data['dewPoint'] = fio_record.data[0].d['dewPoint']
    else:
        weather_data['dewPoint'] = 'unknown'
    weather_data['humidity'] = fio_record.data[0].d['humidity']
    if 'precipType' in fio_record.data[0].d.keys():
        weather_data['precipType'] = fio_record.data[0].d['precipType']
    else:
        weather_data['precipType'] = 'unknown'
    records.insert_one(weather_data)

def get_weather():
    now_time = dt.datetime.now(tz=est)
    base_time = dt.datetime (2013, 5, 1, 12, 0, 0, 0, tzinfo=est)
    while base_time < now_time:
        forecast = forecastio.load_forecast(api_key, lat, lng, time=base_time, units='us')
        daily = forecast.daily()
        insert(base_time, daily)
        print "updated weather ", base_time.date()
        base_time = base_time + dt.timedelta(hours=24)

if __name__ == "__main__":
    get_weather()
S
