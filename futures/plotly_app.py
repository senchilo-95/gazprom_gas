from django_plotly_dash import DjangoDash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import plotly.express as px
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlalchemy as sa
dict_dates = {1:'янв',2:'фев',3:'мар',4:'апр',5:'май',6:'июн',7:'июл',8:'авг',9:'сен',10:'окт',11:'ноя',12:'дек'}
dict_dates_full = {1:'января',2:'февраля',3:'марта',4:'апреля',5:'мая',6:'июня',7:'июля',8:'августа',9:'сентября',10:'октября',11:'ноября',12:'декабря'}

# df = pd.read_csv('dutch_futures.csv',sep=';')
# df=df[['MONTH','CHANGE','SETTLE']]
# df['DT']=pd.date_range(start=datetime.datetime(2022,9,1),end=datetime.datetime(2032,1,1),freq='1M')

engine = sa.create_engine('sqlite:///consum.sqlite3')
connection=engine.connect()

# result = engine.execute("""
#         CREATE TABLE "gas_futures" (
#            date DATETIME,
#            month_name TEXT,
#             change_price FLOAT,
#             settle_price FLOAT,
#            PRIMARY KEY (date)
#         )
#           """)
# df.to_sql('gas_futures', con=connection, index=False, if_exists='replace')
# inspector = sa.inspect(engine)
# schemas = inspector.get_schema_names()
# for schema in schemas:
#     print("schema: %s" % schema)
#     for table_name in inspector.get_table_names(schema=schema):
#         for column in inspector.get_columns(table_name, schema=schema):
#             print("Column: %s" % column)

command=("""
SELECT *
FROM [gas_futures]
""")

df = pd.read_sql_query(command,connection)
df['DT']=pd.to_datetime(df['DT'])


df_train = df[:36]
y_train = df_train['SETTLE'].values
x_train = df_train.index
def func(x,a,b,c):
    return a * np.exp(-b * x**2) + c
# подбор оптимальных параметров
popt, pcov = curve_fit(func, x_train, y_train,
                       method='lm',
                       maxfev=6000)

a_0 = popt[0]
b_0=popt[1]
c_0=popt[2]

#  экспонента
def func(x):
    return (a_0 * np.exp(-b_0 * x**2) + c_0)+20*np.sin(0.5*x)*np.exp(-0.02*x)
# подбор оптимальных параметров
# popt, pcov = curve_fit(func, df.index, df['SETTLE'].values,
#                        method='lm',
#                        maxfev=6000)
# расчет ошибки
mse = np.sqrt(mean_squared_error(y_train,func(x_train)))

futures = dbc.Card([dcc.Graph(id='my-graph')])
cards = html.Div(
    [
        dbc.Card(
            dbc.CardBody([futures]),
            className="mb-3",style={'min-width':'600px'})
    ]
)

slider=dcc.Slider(37, len(df),1, value=51,
    marks={idx: {'label':'{} {}'.format(dict_dates[df['DT'][idx].month], df['DT'][idx].year),
                 'style':{'writing-mode': 'vertical-rl','text-orientation': 'use-glyph-orientation','height':'100px'}
                 } for idx in range(37,len(df),1)},
    included=True,
    id='date_slider'
)

slider_title = html.Div('Выберите месяц, до которого выполняется экстраполяция',style={'text-align':'center'})
app = DjangoDash('SimpleExample',add_bootstrap_links=True)
app.css.append_css({ "external_url" : "/static/futures/css/main.css" })
app.layout =  html.Div([cards,slider_title,slider],style={'background-color': '#D0DBEA','min-width':'600px','width':'100%','height':'700px'})

@app.callback([Output('my-graph', 'figure')],
              [Input('date_slider', 'value')])
def update_graph(dot):
    df_test = df[36:dot]
    y_test = df_test['SETTLE'].values
    x_test = df_test.index
    mse_forecast = np.sqrt(mean_squared_error(y_test, func(x_test)))

    figure = px.scatter(df_train.reset_index(),
                     x='DT',
                     y='SETTLE',
                     template='plotly_white',
                     hover_data={"DT": "|%B %M"},
                     labels={
                         "DT": "Месяц",
                         "SETTLE": "Цена фьючерсного контракта на газ в Голландии, $"
                     },
                     title="Кривая фьючерсов. Среднеквадратичная ошибка для восстановленной кривой = {:.1f}$, для прогноза = {:.1f}$".format(mse,mse_forecast)
                        )

    figure.add_trace(go.Scatter(x=df_train['DT'], y=func(x_train),
                             mode='lines',
                             name='Восстановленная кривая (36 месяцев)'))

    figure.add_trace(go.Scatter(x=df_test['DT'], y=df_test['SETTLE'],
                                mode='markers',
                                name='Цены фьючерсов (будущее)'))
    figure.add_trace(go.Scatter(x=df_test['DT'], y=func(x_test),
                                mode='lines',
                                name='Восстановленная кривая (прогноз)'))

    figure.update_layout(height=500)


    return [figure]



