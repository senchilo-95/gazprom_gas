import sqlalchemy as sa
from django_plotly_dash import DjangoDash
import datetime
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
import numpy as np
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

test_len = 3  # длина тестовой выборки
len_forecast = 12  # горизонт прогноза



def mape(true,preds):
    return np.round(np.mean(np.abs(true-preds)/true)*100,2)

dict_dates = {1:'янв',2:'фев',3:'мар',4:'апр',5:'мая',6:'июн',7:'июл',8:'авг',9:'сен',10:'окт',11:'ноя',12:'дек'}
dict_dates_full = {1:'января',2:'февраля',3:'марта',4:'апреля',5:'мая',6:'июня',7:'июля',8:'августа',9:'сентября',10:'октября',11:'ноября',12:'декабря'}

# toc=eurostat.get_toc_df()
# result=eurostat.subset_toc_df(toc, 'Supply, transformation and consumption of gas - monthly data')
# result_code = str(result.code.values[0])
# df = eurostat.get_data_df(result_code,flags=False)
# countries_list = ['BE','FR','ES']
# target_unit = 'MIO_M3' #нам нужны только объемы - миллионы куб.м.
# code_col = r'geo\time' # признак страны
# feature='IC_CAL_MG' # необходимый признак
# df_sorted=df[(df[code_col].isin(countries_list))&(df['nrg_bal'] == feature)&(df['unit'] == target_unit)]
# df_t = df_sorted[df_sorted.columns[3:]].set_index(code_col).T
# df_t.index=pd.to_datetime(df_t.index,format='%YM%m')
# df_t.dropna(inplace=True) # удаляем пустые данные до 2014 года и за июнь-июль 2022 г.
# df_t.sort_index(inplace=True)
#
# df_melt=pd.melt(df_t,ignore_index=False).reset_index()
# df_melt.columns=['date','country','gas_supply']

engine = sa.create_engine('sqlite:///consum.sqlite3')
connection=engine.connect()

# result = engine.execute("""
#         CREATE TABLE "gas_supply" (
#            date DATETIME,
#            country TEXT,
#             gas_supply FLOAT,
#            PRIMARY KEY (date,country)
#         )
#           """)
#

# df_melt.to_sql('gas_supply', con=connection, index=False, if_exists='replace')


command=("""
SELECT *
FROM [gas_supply]
""")

df_db = pd.read_sql_query(command,connection)
df_db['date']=pd.to_datetime(df_db['date'])
df_t=pd.pivot_table(df_db,index='date',columns='country',values='gas_supply')

# df = pd.read_excel('/Users/nikitasencilo/PycharmProjects/gazp_exp_tst/ogk-web-prices/supply/forecasts.xlsx')
# df=(df[df.columns[1:]])
# result = engine.execute("""
#         CREATE TABLE "supply_forecast" (
#            date DATETIME,
#            country TEXT,
#             forecast FLOAT,
#            PRIMARY KEY (date,country)
#         )
#           """)
#
# df.to_sql('supply_forecast', con=connection, index=False, if_exists='replace')

command=("""
SELECT *
FROM [supply_forecast]
""")

df_forecast = pd.read_sql_query(command,connection)
forecast_pivot = pd.pivot_table(df_forecast,index='date',columns='country',values='forecast')


ues_tabs = dcc.Tabs(
    id="tabs-with-classes",
    value='BE',
    parent_className='custom-tabs',
    className='custom-tabs-container',
    children=[
        dcc.Tab(
            label='Бельгия',
            value='BE', className='custom-tab',
            selected_className='custom-tab--selected'
        ),
        dcc.Tab(
            label='Испания',
            value='ES',
            className='custom-tab',
            selected_className='custom-tab--selected'
        ),
        dcc.Tab(
            label='Франция',
            value='FR',
            className='custom-tab',
            selected_className='custom-tab--selected'
        )
    ])

month_gas = dbc.Card([dcc.Graph(id='my-graph-m')])
year_gas = dbc.Card([dcc.Graph(id='my-graph-y')])
cards = html.Div(
    [
        dbc.Card(
            dbc.CardBody([month_gas]),
            className="mb-3",style={'min-width':'600px'}),
        dbc.Card(
            dbc.CardBody([year_gas]),
            className="mb-3",style={'min-width':'600px'})
    ]
)

countries_dict = {'BE':'Бельгии','FR':'Франции','ES':'Испании'}
app = DjangoDash('GasApp',add_bootstrap_links=True)
app.css.append_css({ "external_url" : "/static/futuresfutures/css/main.css" })
app.layout =  html.Div([ues_tabs,cards],style={'background-color': '#D0DBEA','min-width':'600px','width':'100%'})

@app.callback([Output('my-graph-y', 'figure'),Output('my-graph-m', 'figure')],
              [Input('tabs-with-classes', 'value')])
def update_graph(tab):

    data = pd.DataFrame(df_t[df_t.index.year >= 2019][tab])
    data_test = data[-6:-6 + test_len][tab]

    date_range = [datetime.datetime(dt.year, dt.month, 1) for dt in
                  pd.date_range(start='2022-01', freq='1M', periods=len_forecast)]
    future_for_plot = pd.DatetimeIndex(date_range)
    forecast_country = forecast_pivot[tab]
    mape_fact = np.round(mape(data_test.values, forecast_country[:test_len].values),2)
    mse=np.round(np.sqrt(mean_squared_error(data_test.values, forecast_country[:test_len].values)))


    df = pd.DataFrame(data[data.index.year == 2019])
    df.columns = [2019]
    df['m'] = df.index.month
    df_22 = pd.DataFrame(data[data.index.year == 2022])
    df_22.columns = [2022]
    df_22['m'] = df_22.index.month
    df[2020] = data[data.index.year == 2020].values
    df[2021] = data[data.index.year == 2021].values
    df = df.merge(df_22, left_on=['m'], right_on=['m'], how='outer')
    df.drop(columns='m', inplace=True)
    df['DT'] = data[data.index.year == 2019].index


    figure = px.line(df,
                     x='DT',
                     y=df.columns,
                     template='plotly_white',
                     hover_data={"DT": "|%B %M"},
                     labels={
                         "DT": "Месяц",
                         "value": "Объем потребления газа, млн. куб. м.",
                         'variable': 'Год'
                     },
                     title=f"Сравнение по годам объемов потребления газа в {countries_dict[tab]}. MSE (тест 3 мес.) = {mse} млн.куб.м.({mape_fact}%)",
                     color_discrete_map={'2019': '#6DCFF6', '2020': '#0094D8', '2021': '#18335D',
                                         'Прогноз 2022': '#FF9C00',
                                        '2022': '#D50E2F'},
                     markers=True)

    figure.add_trace(go.Scatter(x=df['DT'], y=forecast_country.values,
                                mode='lines+markers',
                                name='Прогноз 2022'))
    figure.update_xaxes(
        dtick="M1",
        tickformat="%b"
        )
    figure.update_layout(height=500)

    data['rolling_mean'] = data.rolling(3).mean()
    data=data.reset_index()
    data.columns = ['DT','Объем потребления газа','Cкользящее среднее (3 мес)']
    figure2 = px.line(data,
                     x='DT',
                     y=data.columns,
                     template='plotly_white',
                     hover_data={"DT": "|%B %y"},
                     labels={
                         "DT": "Месяц",
                         "value": "Объем потребления газа, млн. куб. м.",
                         'variable': 'Год'
                     },
                     title=f"Объем потребления газа в {countries_dict[tab]}. MSE (тест 3 мес.) = {mse} млн.куб.м.({mape_fact}%)",
                     color_discrete_map={'Объем потребления газа': '#6DCFF6',
                                         'Cкользящее среднее (3 мес)': '#FF9C00'},
                     markers=True)
    figure2.add_trace(go.Scatter(x=future_for_plot, y=forecast_country.values,
                                mode='lines+markers',
                                name='Прогноз 2022'))
    figure2.update_xaxes(
        dtick="M1",
        tickformat="%b"
    )
    figure2.add_vrect(x0=data['DT'][0], x1=data['DT'][0],
                     annotation_text='2019', annotation_position="top left",
                     fillcolor="#8DB1E1", opacity=0.2, line_width=4)
    figure2.add_vrect(x0=data['DT'][12], x1=data['DT'][12],
                      annotation_text='2020', annotation_position="top left",
                      fillcolor="#8DB1E1", opacity=0.2, line_width=4)
    figure2.add_vrect(x0=data['DT'][24], x1=data['DT'][24],
                      annotation_text='2021', annotation_position="top left",
                      fillcolor="#8DB1E1", opacity=0.2, line_width=4)
    figure2.add_vrect(x0=data['DT'][36], x1=data['DT'][36],
                      annotation_text='2022', annotation_position="top left",
                      fillcolor="#8DB1E1", opacity=0.2, line_width=4)
    figure2.update_layout(height=500)
    return figure,figure2



