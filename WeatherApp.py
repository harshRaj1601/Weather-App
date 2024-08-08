import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import requests
from datetime import datetime
import base64
import pytz

st.set_page_config(
    page_title="Weather Forecast App",
    page_icon="🌡️",
    layout="wide"
)

api_key = "49bd67e0b3eb666fba5b2548eec4f405"
base_url = "http://api.openweathermap.org/data/2.5/"

def cityweather(city_name, type="weather", unit="metric"):
    complete_url = f"{base_url}{type}?appid={api_key}&q={city_name}&units={unit}"
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        return sort_data(x)
    else:
        return sort_data(0)

def sort_data(weather_data):
    
    """"Function to extract data from json"""
    if weather_data:
        extracted_data = []
        try:
            for i in range(len(weather_data['list'])):
                date, time = weather_data['list'][i]['dt_txt'].split(' ')
                act_temp = weather_data['list'][i]['main']['temp'] 
                feel_temp = weather_data['list'][i]['main']['feels_like'] 
                min_temp = weather_data['list'][i]['main']['temp_min'] 
                max_temp = weather_data['list'][i]['main']['temp_max'] 
                pressure = weather_data['list'][i]['main']['pressure'] 
                sea_level = weather_data['list'][i]['main']['sea_level']
                grnd_level = weather_data['list'][i]['main']['grnd_level']
                humidity = weather_data['list'][i]['main']['humidity']
                weather_status = weather_data['list'][i]['weather'][0]['main']
                extracted_data.append((weather_data['city']['name'],weather_data['city']['country'],date, time, act_temp, feel_temp, 
                                    min_temp, max_temp, pressure, sea_level, grnd_level,humidity, weather_status))
            df = pd.DataFrame(extracted_data)
            df.rename(columns={0:'city',1:'country',2: 'date', 3:'time',4:'actual_temp',5:'feels_like_temp',6:'min_temp',7: 'max_temp',
                       8:'pressure', 9: 'sea_level', 10: 'grnd_level', 11: 'humidity', 12: 'weather_desc'},inplace=True)
            df["datetime"] = df["date"]+" "+df["time"]
            return df
        except:
            d = dict(
                lat=weather_data["coord"]["lat"],
                lon=weather_data["coord"]["lon"],
                weather_id=weather_data["weather"][0]["id"],
                weather_main=weather_data["weather"][0]["main"],
                weather_description=weather_data["weather"][0]["description"],
                weather_icon=weather_data["weather"][0]["icon"],
                temp=weather_data["main"]["temp"],
                feels_like=weather_data["main"]["feels_like"],
                temp_min=weather_data["main"]["temp_min"],
                temp_max=weather_data["main"]["temp_max"],
                pressure=weather_data["main"]["pressure"],
                humidity=weather_data["main"]["humidity"],
                visibility=weather_data["visibility"],
                wind_speed=weather_data["wind"]["speed"],
                wind_deg=weather_data["wind"]["deg"],
                clouds_all=weather_data["clouds"]["all"],
                dt=datetime.fromtimestamp(weather_data["dt"]).strftime("%Y-%m-%d %H:%M:%S"),
                sunrise=weather_data["sys"]["sunrise"],
                sunset=weather_data["sys"]["sunset"],
                timezone=weather_data["timezone"],
                id=weather_data["id"],
                name=weather_data["name"],
                cod=weather_data["cod"]
            )
           #df = pd.DataFrame(d,index=[0])
            return d
    else:
        return 0
    
def sun_location(sr,ss,timezone):
    sunrise = datetime.fromtimestamp(sr)
    sunset = datetime.fromtimestamp(ss)
    r = sunrise.strftime('%H:%M')
    s = sunset.strftime('%H:%M')
    now = datetime.now(pytz.utc)
    target_timezone = pytz.FixedOffset(timezone//60)
    target_time = now.astimezone(target_timezone)
    ct = target_time.strftime('%H:%M')
    current_time = datetime.strptime(ct,'%H:%M')
    total_daylight = (sunset - sunrise).seconds
    elapsed_time = (current_time - sunrise).seconds
    angle = (elapsed_time / total_daylight) * np.pi  # angle in radians
    x_sun = 1 + np.cos(angle)
    y_sun = 1 + np.sin(angle)
    fig = go.Figure()
    theta = np.linspace(0, 2*np.pi, 100)
    x = 1 + np.cos(theta)
    y = 1 + np.sin(theta)
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color="#111111", width=3)))
    fig.add_trace(go.Scatter(x=[2-x_sun] if timezone>0 else [x_sun], y=[y_sun] if timezone>0 else [2-y_sun],
                             mode='text',text="🔆",textfont=dict(size=30), textposition = "middle center", name="Current Position"))
    fig.add_trace(go.Scatter(x=[0, 2], y=[1, 1],
                            mode='markers+text',
                            marker=dict(size=12, color='yellow'),
                            text=[r, s] if timezone > 0 else [s,r],
                            textposition='bottom center'))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 2.5],autorange=True),
        yaxis=dict(visible=False, range=[0, 2],autorange=True),
        showlegend=False,
    )

    st.plotly_chart(fig,use_container_width=True)

def format_timezone_offset(offset_seconds):
    hours = offset_seconds // 3600
    minutes = abs(offset_seconds % 3600) // 60
    sign = '+' if hours >= 0 else '-'
    return f"{sign}{abs(hours):02}:{minutes:02}"


def get_current_time(times):
    now = datetime.now(pytz.utc)
    target_timezone = pytz.FixedOffset(times//60)
    target_time = now.astimezone(target_timezone)
    formatted_time = target_time.strftime('%Y-%m-%d %H:%M:%S')
    date_obj = datetime.strptime(formatted_time, "%Y-%m-%d %H:%M:%S")
    date = date_obj.strftime("%d %B, %Y")
    time = date_obj.strftime("%I:%M %p").lower()
    return date,time

@st.cache_data
def load_city_data():
    return pd.read_csv("cities.csv")

with st.spinner("Loading dataset..."):
    df = load_city_data()

countries = df['country_name'].unique().tolist()
country = st.sidebar.selectbox("Select a Country", countries,index=countries.index("India"))
states = df[df["country_name"] == country]["state_name"].unique().tolist()
state = st.sidebar.selectbox("Select a State",states,index=states.index("Uttar Pradesh") if country == "India" else 0 )
cities = df[df["state_name"] == state]["name"].tolist()
city = st.sidebar.selectbox("Select a City",cities,index=cities.index("Lucknow") if state == "Uttar Pradesh" else 0)
if "District" in city:
    city = city.replace(" District", "")
data = cityweather(f"{city}, {df[df['country_name'] == country]['country_code'].unique()[0]}")
data_frame = cityweather(f"{city}, {df[df['country_name'] == country]['country_code'].unique()[0]}","forecast")


st.title("Weather Forecasting App🌡️")
st.text('Get the current weather for your city, plus a 5-day forecast.')
st.divider() 
with st.container():
    def get_image_base64(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()    
    if data:
        path = f"images/{data['weather_icon']}.png"
        st.markdown(f"""## {city}, {country} ![Image](data:image/png;base64,{get_image_base64(path)})""")
        
        lat = data['lat']
        lon = data['lon']
        n = e = True
        if lat < 0:
            lat *= -1
            n = False

        if lon < 0:
            lon *= -1
            e = False
        st.sidebar.divider()
        st.sidebar.text(f"Location: {lat}°{'N' if n else 'S'} and {lon}°{'E' if e else 'W'}")
        date, time = get_current_time(data["timezone"])
        try:
            temp_delta = float(data["temp"]) - float(data_frame["actual_temp"].mean())
            feel_delta = float(data["feels_like"]) - float(data_frame["feels_like_temp"].mean())
        except:
            temp_delta = 0
            feel_delta = 0
        
        # Display weather information
        col1,col2 = st.columns(2)
        col1.metric("Temperature", f"{round(data['temp'],3)} °C",delta=f"{temp_delta:.2f} °C",delta_color="inverse")
        col2.metric("Feels Like", f"{round(data['feels_like'],3)} °C",delta=f"{feel_delta:.2f} °C",delta_color="inverse")
        
        
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
            col1.metric("Humidity", f"{data['humidity']} %")
            col1.metric("Pressure", f"{data['pressure']} hPa")
            col1.metric("Visibility", f"{data["visibility"]//1000} Km")
                     
        with col2:
            col2.metric("Status", f"{data['weather_description'].capitalize()}")
            col2.metric(f"Time ({format_timezone_offset(data["timezone"])} UTC)",f"{time}")
            col2.metric("Wind Speed",f"{data["wind_speed"]} m/sec")
        with col3:
            col3.metric("Clouds",f"{data["clouds_all"]} %")
            col3.metric("Date",f"{date}")
            col3.metric("Wind Direction",f"{data["wind_deg"]} °")
            
        fig1 = px.line(data_frame,x="datetime",y=["actual_temp","feels_like_temp"],title='Temperature Forecast for next 5 days')
        st.plotly_chart(fig1,use_container_width=True)
        
        fig = px.scatter(title='Minimum and Maximum Temperature')
        fig.add_scatter(x=data_frame['date'].unique() ,y=data_frame.groupby('date')['max_temp'].max(),name='Maximum Temperature')
        fig.add_scatter(x=data_frame['date'].unique() ,y=data_frame.groupby('date')['min_temp'].min(),name='Minimum Temperature')
        fig.update_yaxes(title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
        sun_location(data["sunrise"],data["sunset"],data["timezone"])
            
    else:
        st.error(f"Data of {city} is not available, Sorry!")
