import ephem
# import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from itertools import cycle
from math import *
import datetime
from scipy import integrate
from scipy import signal
import numpy
import pendulum
from timezonefinder import TimezoneFinder
import pickle
import sys
from decimal import Decimal
import line_profiler



def plot_sun_path():
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)

    fig1, ax1 = plt.subplots(1,figsize=(10,8))
    legend_labels = []

    sun = ephem.Sun()
    place = ephem.Observer()
    place.lon, place.lat = '-122.1029729', '37.5332387'
    #latLong = {"lat":37.5332387,"long":-122.1029729}

    sun_alt = []
    sun_az = []

    place.date = "2018/1/15 8:00"

    for d in range(1,13):
        for t in range(1440):
            sun.compute(place)
            if ephem.degrees(sun.alt) > 0:
                alt_deg = ephem.degrees(sun.alt.norm) * 57.2957795131
                sun_alt.append(alt_deg)
                az_deg = ephem.degrees(sun.az.norm) * 57.2957795131
                sun_az.append(az_deg)

            place.date = place.date + ephem.minute
            #print("altitude: %s, azimuth: %s" % (sun.alt, sun.az))

        ax1.plot(sun_az,sun_alt,next(linecycler),linewidth=2)
        legend_labels.append(d)

        place.date = "2018/%s/11 8:00" % (d+1)
        sun_alt = []
        sun_az = []

    #ax1.set_xlim(360,540)
    ax1.legend(labels = legend_labels,loc='upper right')
    ax1.set_xlabel("Azimuth [N clockwise deg]")
    ax1.set_ylabel("Altitude [deg]")
    #plt.savefig('graph of sun position per month.png', dpi = 600)
    plt.show()

#@profile
def hourAngle(obs,utc_offset,time):
    local_time = time

    if local_time.is_dst():
        DST = 1
    else:
        DST = 0

    day_angle = dayAngle(obs,utc_offset,local_time)

    E_t = 229.18*(0.000075+0.001868*cos(day_angle)-0.032077*sin(day_angle)-0.014615*cos(2*day_angle)-0.04089*sin(2*day_angle))
    long_correction = 4*(degrees(obs.lon)-utc_offset*15) #closest standard meridian (non DST)
    solar_time =  local_time.add(hours=-DST, minutes=long_correction+E_t)
    st = datetime.timedelta(hours=solar_time.hour, minutes =solar_time.minute, seconds = solar_time.second).total_seconds()/3600
    # print("solar_time",solar_time)
    hour_angle = (st-12)*15*pi/180 #in rads

    return hour_angle, st,solar_time # radians

#@profile
def dayAngle(obs,utc_offset,time):
    local_time = time
    d_n = local_time.day_of_year
    day_angle = 2*pi*(d_n-1)/365

    return day_angle

#@profile
def sunsetHourAngle(sun,obs,beta): #towards equator
    sun.compute(obs)
    beta = 0.0174533 * beta
    hour1 = acos(-tan(sun.dec)*tan(obs.lat))
    hour2 = acos(-tan(sun.dec)*tan(obs.lat-beta))

    return min(hour1,hour2) #radians

#@profile
def direct_beam_ext(sun,obs,beta,t_i,t_f,utc_offset,time):
    I_sc = 4921#kJ m^-2 hr-1 ... 1367 W m^-2
    beta = 0.0174533 * beta # deg -> rad

    sun.compute(obs)

    day_angle = dayAngle(obs,utc_offset,time)

    E_0 = 1.00011+0.034221*cos(day_angle)+0.00128*sin(day_angle)+0.000719*cos(2*day_angle)+0.000077*sin(2*day_angle)

    I_0b = I_sc*E_0*12/pi*(-cos(sun.dec)*cos(beta - obs.lat)*(sin(t_i) - sin(t_f)) + (t_i - t_f)*sin(sun.dec)*sin(beta - obs.lat))

    return I_0b #kJ m^-2

#@profile
def clearness_index_year(data_kt):
    date = next(iter(data_kt))
    base_year = date.year

    return base_year

#@profile
def clearness_index(time,data_kt):
    t = hour_rounder(time).in_tz('UTC')
    # print(t)
    if len(data_kt[t]) == 0:
        while len(data_kt[t])==0:
            t = t.add(hours=1)
        k_t_list = data_kt[t]
    else:    
        k_t_list = data_kt[t]

    return k_t_list

#@profile
def hour_rounder(t):
    # Truncates to base hour
    rounded_t = t.set(second=0, microsecond=0, minute=0, hour=t.hour)
    return rounded_t

#@profile
def insolation_modified_atmosphere(surface_albedo,beta,place,sun,utc_offset,wi,wf,time):
    # time = pendulum.instance(place.date.datetime())
    # print(time)

    sun.compute(place)

    I_h_extra = direct_beam_ext(sun,place,0,wi,wf,utc_offset,time)

    k_t_list = clearness_index(time.set(year=1991),data_kt)

    # print(k_t_list)

    k_t = numpy.mean(k_t_list)

    if k_t > 0.8: #limit to practical maximum
        k_t = 0.8
    
    I_h = I_h_extra*k_t #GHI
    # print(k_t)

    if (k_t >=0 and k_t <=0.35):
        I_dh = I_h * (1.0-0.249*k_t) #DHI
    elif (k_t > 0.35 and k_t <=0.75):
        I_dh = I_h * (1.557-1.84*k_t)
    elif (k_t > 0.75):
        I_dh = I_h * 0.177

    I_b = I_h - I_dh #DNI

    beta = 0.0174533 * beta #convert to rads

    azimuth = sun.az - pi #change from N to S origin

    cos_incidence_angle = cos(sun.alt)*cos(azimuth-0)*sin(beta)+sin(sun.alt)*cos(beta) #rads, 0 amizuth orientation
    # cos_incidence_angle_alternate = sin(sun.dec)*sin(place.lat-beta) + cos(sun.dec)*cos(place.lat-beta)*cos(wi)
    # print("cos incidence_angle",cos_incidence_angle)
    # print("cos incidence_angle_IBQUAL",cos_incidence_angle_alternate)
    # print("")
    # print("sin sunalt",sin(sun.alt))
    R_b = cos_incidence_angle/sin(sun.alt)
    R_b_alternate = 0#cos_incidence_angle_alternate/sin(sun.alt)
    # print("ih",I_h)
    # print("Ib",I_b)
    # print("I_dh",I_dh)
    if R_b <0:
        # print("r_b",R_b)
        # print("cos incidence_angle",cos_incidence_angle)
        # print(time)
        R_b = 0

    # if R_b >15:
    #     print(time)
    #     print("r_b",R_b)
    #     print("incidence_angle",acos(cos_incidence_angle)*57.2958)
    #     print("hour angle",wi*57.2958)
    #     print(sin(sun.alt))
    #     print("alt:",sun.alt)
    #     print("")

    I_total = I_b*R_b + I_dh*(1+cos(beta))/2 + surface_albedo*I_h*(1-cos(beta))/2
    # print(I_total)

    return I_total,R_b,R_b_alternate

#@profile
def daily_insolation(time,place,utc_offset,sun,b,surface_albedo): #b in deg
    sun_insol = []
    sun_insol_atmosphere = []
    time_hours = []
    # R_b_collector = [[],[]]

    start_time = time.set(second=0, microsecond=0, minute=0, hour=0)
    place.date = start_time.in_tz('UTC')

    # print("local start time: ",start_time)
    # print("UTC place time: ",place.date)

    sunset = sunsetHourAngle(sun,place,b)
    # print("sunset angle:", sunset*57.2958)

    for l in range(288):

        sun.compute(place)

        wi,st,solar_time = hourAngle(place,utc_offset,start_time)

        if (wi>(-sunset+pi/180) and wi<(sunset-pi/180) and sun.alt>(pi/180)):
            # print("hour angle",wi/0.0174533,"deg")
            start_time = start_time.add(minutes=5)
            place.date = start_time.in_tz('UTC')

            wf,st,solar_time = hourAngle(place,utc_offset,start_time)

            start_time = start_time.subtract(minutes=5)
            place.date = start_time.in_tz('UTC')

            sun_insol.append(direct_beam_ext(sun,place,b,wi,wf,utc_offset,start_time))

            I_total, R_b, R_b_alternate = insolation_modified_atmosphere(surface_albedo,b,place,sun,utc_offset,wi,wf,start_time)

            # R_b_collector[0].append(R_b)
            # R_b_collector[1].append(R_b_alternate)

            sun_insol_atmosphere.append(I_total)
            
            time_hours.append(l/12)
        # elif (wi>(-sunset+pi/180) and wi<(sunset-pi/180)):
        #     print(start_time)

        start_time = start_time.add(minutes=5)
        place.date = start_time.in_tz('UTC')

    place.date = start_time.subtract(minutes=288).in_tz('UTC')

    daily_irradiation = integrate.simps(sun_insol)
    daily_irradiation_atmosphere = integrate.simps(sun_insol_atmosphere)

    return time_hours, st, daily_irradiation, sun_insol, sun_insol_atmosphere,daily_irradiation_atmosphere #kJ/m^2

def plot_insolation_yearly(beta,longitude,latitude,continuous,surface_albedo):

    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    tf = TimezoneFinder()
    
    fig2, ax2 = plt.subplots(1,figsize=(10,8))

    legend_labels = []

    sun = ephem.Sun()
    place = ephem.Observer()
    place.lon, place.lat = str(longitude), str(latitude)

    sun_insol = []
    time_series = []
    solar_time_series = []
    daily_irradiation = []
    sun_insol_atmosphere_perDay = []

    year = 2013
    time_zone = tf.timezone_at(lng=longitude, lat=latitude)
    utc_offset = pendulum.from_timestamp(0, time_zone).offset_hours

    for count,b in enumerate(beta):
        if continuous:
            start_time = pendulum.datetime(year,1,1,0,0,0,tz=time_zone)
            for i in range(1,365):
                time_hours, st, daily_irradiation_value ,null,null,daily_irradiation_atmosphere= daily_insolation(start_time,place,utc_offset,sun,b,surface_albedo)
                daily_irradiation.append(daily_irradiation_value)
                sun_insol_atmosphere_perDay.append(daily_irradiation_atmosphere)
                start_time = start_time.add(days=1)
                progressBar(i, 365, bar_length=20)

                # if (i<300 and i >293):
                #     time_hours, st, daily_irradiation_value ,null,null,daily_irradiation_atmosphere = daily_insolation(start_time,place,utc_offset,sun,b,surface_albedo)
                #     print(i, daily_irradiation_atmosphere)


            savgol_smoothed = signal.savgol_filter(sun_insol_atmosphere_perDay, 101,5)
            ax2.plot(range(1,365),daily_irradiation,c=line_colors[count],label='Angle: {0}'.format(b))
            ax2.plot(range(1,365),sun_insol_atmosphere_perDay,':',c=line_colors[count])
            ax2.plot(range(1,365),savgol_smoothed,'--',c=line_colors[count])
            print("")
            print(b,' degrees (extra total): ',format_e(integrate.simps(daily_irradiation)), "kJ m-2")
            print(b,' degrees (atm raw total): ',format_e(integrate.simps(sun_insol_atmosphere_perDay)),"kJ m-2")
            print(b,' degrees (atm savgol total): ',format_e(integrate.simps(savgol_smoothed)),"kJ m-2")
            # current_label = 'Angle: {0}'.format(b)
            # legend_labels.append(current_label)
            
            daily_irradiation = []
            sun_insol_atmosphere_perDay =[]

        if continuous:
            ax2.legend()#(labels = legend_labels,loc='upper right')
            ax2.set_xlabel("Day Number")
            ax2.set_ylabel("Irradiation [kJ/m^2]")
        #plt.savefig('graph of sun position per month.png', dpi = 600)
    plt.show()

def find_optimum_insolation_yearly(longitude,latitude,continuous,surface_albedo):
    
    tf = TimezoneFinder()
    
    fig2, ax2 = plt.subplots(3, sharex=True,figsize=(10,12))

    legend_labels = []

    sun = ephem.Sun()
    place = ephem.Observer()
    place.lon, place.lat = str(longitude), str(latitude)

    sun_insol = []
    time_series = []
    solar_time_series = []
    daily_irradiation = []
    sun_insol_atmosphere_perDay = []

    beta_logger = []
    sd_logger=[]
    minimum_yearly_logger=[]

    beta_logger_atm = []
    sd_logger_atm=[]
    minimum_yearly_logger_atm=[]

    year = 2013
    time_zone = tf.timezone_at(lng=longitude, lat=latitude)
    utc_offset = pendulum.from_timestamp(0, time_zone).offset_hours

    beta_range = range(25,int(latitude+10))

    for b in beta_range:
        if continuous:
            time = pendulum.datetime(year,1,1,0,0,0,tz=time_zone)
            place.date = time.in_tz('UTC') #place.date.datetime() - datetime.timedelta(hours=utc_offset)

            for i in range(1,365):
                time_hours, st, daily_irradiation_value ,null,null,daily_irradiation_atmosphere = daily_insolation(time,place,utc_offset,sun,b,surface_albedo)
                daily_irradiation.append(daily_irradiation_value)
                sun_insol_atmosphere_perDay.append(daily_irradiation_atmosphere)

                time = time.add(days=1)
                place.date = time.in_tz('UTC')
                progressBar(i, 365, bar_length=20)
                # place.date += 1

            total_yearly_irradiation = numpy.mean(daily_irradiation)
            yearly_deviation = numpy.std(daily_irradiation)
            minimum_yearly = numpy.amin(daily_irradiation)

            smoothed_atm = signal.savgol_filter(sun_insol_atmosphere_perDay, 101,3)
            total_yearly_irradiation_atm = numpy.mean(smoothed_atm)
            yearly_deviation_atm = numpy.std(smoothed_atm)
            minimum_yearly_atm = numpy.amin(smoothed_atm)

            daily_irradiation = []
            sun_insol_atmosphere_perDay =[]

            print(b,' degrees: ',total_yearly_irradiation, " sd = ",yearly_deviation, "min = ",minimum_yearly)
            print(b,' degrees: ',total_yearly_irradiation_atm, " sd = ",yearly_deviation_atm, "min = ",minimum_yearly_atm)
            print("")

            beta_logger.append(total_yearly_irradiation)
            sd_logger.append(yearly_deviation)
            minimum_yearly_logger.append(minimum_yearly)

            beta_logger_atm.append(total_yearly_irradiation_atm)
            sd_logger_atm.append(yearly_deviation_atm)
            minimum_yearly_logger_atm.append(minimum_yearly_atm)
        progressBar(b, len(beta_range), bar_length=20)

    ax2[0].plot(beta_range,beta_logger,'r')
    ax2[0].plot(beta_range,beta_logger_atm,'r--')
    ax2[0].set_ylabel("Irradiation per year [kJ/m^2/yr]")

    ax2[1].plot(beta_range,sd_logger,'g')
    ax2[1].plot(beta_range,sd_logger_atm,'g--')
    ax2[1].set_ylabel("Standard Deviation [kJ/m^2/yr]")

    ax2[2].plot(beta_range,minimum_yearly_logger,'b')
    ax2[2].plot(beta_range,minimum_yearly_logger_atm,'b--')
    ax2[2].set_ylabel("Min. irradiation per year [kJ/m^2/yr]")
    ax2[2].set_xlabel("Beta angle [deg]")


    plt.show()

def plot_insolation_hourly(beta,longitude,latitude,surface_albedo):
    lines = ["-","--","-.",":"]
    linecycler = cycle(lines)
    month_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    tf = TimezoneFinder()
    
    fig2, ax2 = plt.subplots(1,figsize=(10,8))

    legend_labels = []

    sun = ephem.Sun()
    place = ephem.Observer()
    place.lon, place.lat = str(longitude), str(latitude)

    sun_insol = []
    time_series = []
    solar_time_series = []
    daily_irradiation = []

    year = 2013
    day = 29

    time_zone = tf.timezone_at(lng=longitude, lat=latitude)

    utc_offset = pendulum.from_timestamp(0, time_zone).offset_hours

    for b in beta:
        for month in range(6,7):
            time = pendulum.datetime(year,month,day,0,0,0,tz=time_zone)
            place.date = time.in_tz('UTC')
            time_hours, st, daily_irradiation_value ,sun_insol,sun_insol_atmosphere,daily_irradiation_atmosphere = daily_insolation(time,place,utc_offset,sun,b,surface_albedo)
            

            top = max(sun_insol)
            sun_insol = [x/top for x in sun_insol]

            ax2.plot(time_hours,sun_insol,next(linecycler))

            # ax2.plot(time_hours,R_b_collector[0],next(linecycler))
            # ax2.plot(time_hours,R_b_collector[1],next(linecycler))

            # time = pendulum.datetime(year,month,3,0,0,0,tz=time_zone)
            # place.date = time.in_tz('UTC')
            # time_hours, st, daily_irradiation_value ,sun_insol,sun_insol_atmosphere,daily_irradiation_atmosphere = daily_insolation(time,place,utc_offset,sun,b,surface_albedo)
            # ax2.plot(time_hours,sun_insol,next(linecycler))

            current_label = 'Month: {1}, Angle: {0}'.format(b,month_list[month-1])
            legend_labels.append(current_label)
    
    top = max(sun_insol)
    # minor_ticks = numpy.arange(0, 25, 24)
    # ax2.set_xticks(minor_ticks, minor=True)
    ax2.grid(which='both')    
    ax2.legend(labels = legend_labels,loc='upper right')
    ax2.set_xlabel("Time [hrs from midnight]")
    ax2.set_ylabel("Irradiation [kJ/m^2]")
    #plt.savefig('graph of sun position per month.png', dpi = 600)
    plt.show()

def progressBar(value, endvalue, bar_length=20):

        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def format_e(n):
    return '{0:.4E}'.format(Decimal(n))

global data_kt
print('LOADING')
with open('k_t_dict.pickle', 'rb') as filehandle:  
    # read the data as binary data stream
    data_kt = pickle.load(filehandle)

# longitude = -118.6919205
# latitude = 34.05
# tf = TimezoneFinder()
# t_zone =tf.timezone_at(lng=longitude, lat=latitude)
# utc_offset = pendulum.from_timestamp(0, t_zone).offset_hours
# print("utc offset: ", utc_offset)

# time = pendulum.datetime(2013,11,2,4,0,0, tz=t_zone)

# sun = ephem.Sun()
# place =  ephem.Observer()
# place.lon, place.lat = str(longitude), str(latitude)
# place.date = time.in_tz('UTC') #datetime.datetime(2015, 6, 15, 19, 0, 0)

# albedo = 0.2
# time_hours, st, daily_irradiation, sun_insol, sun_insol_atmosphere,daily_irradiation_atmosphere = daily_insolation(time,place,utc_offset,sun,34.05,albedo)
# print("daily irradiation: ",daily_irradiation)
# print("daily irradiation atm: ",daily_irradiation_atmosphere)

# time = pendulum.datetime(2013,11,3,4,0,0, tz=t_zone)

# time_hours, st, daily_irradiation, sun_insol, sun_insol_atmosphere,daily_irradiation_atmosphere = daily_insolation(time,place,utc_offset,sun,34.05,albedo)
# print("daily irradiation: ",daily_irradiation)
# print("daily irradiation atm: ",daily_irradiation_atmosphere)


# plot_insolation_yearly([30,37],-121.9886, 37.5483,True,0.2)
plot_insolation_hourly([30],-121.9886, 37.5483,0.2)
# find_optimum_insolation_yearly(-121.9886, 37.5483,True,0.2)
#34.05