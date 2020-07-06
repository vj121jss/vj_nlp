import pandas as pd
import numpy as np
from math import pi
from bokeh.models import ColumnDataSource,LabelSet
from bokeh.palettes import Spectral6, Category20c,Category10,Accent,Category20
from bokeh.palettes import Greys256,Inferno256,Magma256,Plasma256,Viridis256,Cividis256,Turbo256,linear_palette,Set3,Pastel1,Spectral5,Spectral6
from bokeh.io import show, output_file, output_notebook, push_notebook
from bokeh.layouts import row, column, gridplot
from bokeh.models import CustomJS, Slider
from bokeh.models.widgets import Div
from bokeh.transform import factor_cmap, cumsum
from bokeh.models.widgets import Paragraph,DataTable, TableColumn
from bokeh.models import ColumnDataSource,HoverTool
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models.layouts import LayoutDOM, Box, Row, Column, GridBox, Spacer, WidgetBox
from bokeh.layouts import widgetbox, column, row
import nltk
from bs4 import BeautifulSoup
import string
import re
from collections import Counter
from nltk import tokenize 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer 
import matplotlib.pyplot as plt
from wordcloud import WordCloud



df=pd.read_csv('train.tsv',delimiter='\t',encoding='utf-8')
df.dropna(inplace=True)
def LableFunc(Sentiment):
    if Sentiment>=3:
        return 'Positive'
    elif Sentiment<=1:
        return 'Negative'
    else:
        return 'Neutral'
df['Label']=df['Sentiment'].apply(LableFunc)
stop_words=set(stopwords.words('english'))
lemma=WordNetLemmatizer()
def clean_Phrase(phrase_text):
    phrase_text=re.sub(r'http\S+',' ',phrase_text)                             #removing the url
    phrase_text=re.sub('[^a-zA-Z]',' ',phrase_text)                            #removing Numbers and punctuatino
    phrase_text=str(phrase_text).lower()                                      #Convert all characters into lowercase
    phrase_text=word_tokenize(phrase_text)                                    #Tokenization
    phrase_text=[item for item in phrase_text if item not in stop_words]      #Removing Stop Words
    phrase_text=[lemma.lemmatize(word=w,pos='v') for w in phrase_text]        #Lemmatization
    phrase_text=[i for i in phrase_text if len(i)>2]                          #Remove the words havig length <=2
    phrase_text=' '.join(phrase_text)                                          #Converting list to string
    return phrase_text

df['clean_Phrase']=df['Phrase'].apply(clean_Phrase)

df_test=pd.read_csv('new_test.tsv',delimiter=',',encoding='utf-8')
df_test.drop('Unnamed: 0',axis=1, inplace=True)
df_test.dropna(inplace=True)

train_gr = df_test.groupby('Sentiment')['Phrase'].apply(list)
train_gr2 = df_test.groupby('Sentiment')['PhraseId'].apply(list)
train_gr3 = df_test.groupby('Sentiment')['SentenceId'].apply(list)
train_gr = pd.DataFrame(train_gr)
train_gr2 = pd.DataFrame(train_gr2)
train_gr3 = pd.DataFrame(train_gr3)
a = pd.concat([train_gr2,train_gr3,train_gr],axis=1)
a= a.reset_index()
b = a.set_index('Sentiment').PhraseId.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'PhraseId'})
c = a.SentenceId.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'SentenceId'})
d = a.Phrase.apply(pd.Series).stack().reset_index(level=0).rename(columns={0:'Phrase'})
new_explode = pd.concat([b,c,d], axis=1)
new_explode.drop('level_0',axis=1,inplace=True)

df_test['clean_Phrase']=df_test['Phrase'].apply(clean_Phrase)

#output_file("NLP_Dashboard_slider.html")
tools="pan,wheel_zoom,box_zoom,reset,save,box_select,hover"

x = Counter({
    'train': 156060,
    'test': 66292,
    'dev': 0
})
d = pd.DataFrame.from_dict(dict(x), orient='index').reset_index().rename(index=str, columns={0:'value', 'index':'data'})
d['percent'] = d['value'] / sum(x.values()) * 100
d['angle'] = d['value'] / sum(x.values()) * 2*pi
d['color'] = Accent[len(x)]
#z=110*(data['value']/data['value'].sum())
p2 = figure(plot_height=400,plot_width=425, title="Count of train and test",
           tools="hover", tooltips="@data:@value", x_range=(-0.5, 0.5))
p2.annular_wedge(x=0, y=1, inner_radius=0.10, outer_radius=0.25, direction="anticlock",
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', source=d)

#labels = LabelSet(x=0, y=1, text='value',
        #angle=cumsum('angle', include_zero=True), source=source, render_mode='canvas')

#p2.add_layout(labels)
p2.axis.axis_label=None
p2.axis.visible=False
p2.grid.grid_line_color = None


x=df['Sentiment'].value_counts()
data = pd.Series(x).reset_index(name='Count').rename(columns={'index':'Sentiment'})
#p4 = figure(tools="pan,wheel_zoom,box_zoom,reset")
p4 = figure(plot_width=440, plot_height=400,x_axis_label="Count of each sentiments", y_axis_label="Count",title="Count of Each Sentiments", tools=tools,toolbar_location="right" ,tooltips=[("Sentiment","@x"),("Count","@top")])
p4.vbar(x=range(5), top=data['Count'], width=0.7,color=Spectral5,line_color="black")
p4.y_range.start =0
#p4.y_range.end = 100000
#p4.xaxis.major_label_orientation = pi/4
p4.grid.grid_line_color = None

train_new = df_test
#train_new.drop(['PhraseId', 'SentenceId'], axis=1, inplace=True)
train_grouped = train_new.groupby('Sentiment')['clean_Phrase'].apply(list)
train_grouped = pd.DataFrame(train_grouped)

def remove_punc(text):
    no_punc = "".join([word for word in text if word not in string.punctuation])
    return no_punc

tokenizer = RegexpTokenizer(r'\w+')

from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
def word_lemma(text):
    lem_text = ' '.join([lemmatizer.lemmatize(i) for i in text])
    return lem_text

train_grouped['clean_Phrase'] = train_grouped['clean_Phrase'].astype('object')
train_grouped['clean_Phrase'] = train_grouped['clean_Phrase'].apply(lambda x : remove_punc(x))
train_grouped['clean_Phrase'] = train_grouped['clean_Phrase'].apply(lambda x : tokenizer.tokenize(x.lower()))
train_grouped['clean_Phrase'] = train_grouped['clean_Phrase'].apply(lambda x : word_lemma(x))


for i in range(0,5):
    data = train_grouped['clean_Phrase'][i]
    cloud = WordCloud(background_color = "black")
    cloud.generate(data)
    cloud.to_file("Sentiment_test_cp"+str(i)+".png")

p30 = figure(plot_width=260, plot_height=300, title="WORST(0)")
p30.image_url(url=["static/image/wc0.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p30.xaxis.visible = None
p30.yaxis.visible = None

p31 = figure(plot_width=260, plot_height=300, title="BAD(1)")
p31.image_url(url=["static/image/wc1.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p31.xaxis.visible = None
p31.yaxis.visible = None

p32 = figure(plot_width=260, plot_height=300, title="AVERAGE(2)")
p32.image_url(url=["static/image/wc2.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p32.xaxis.visible = None
p32.yaxis.visible = None

p33 = figure(plot_width=260, plot_height=300, title="GOOD(3)")
p33.image_url(url=["static/image/wc3.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p33.xaxis.visible = None
p33.yaxis.visible = None

p34 = figure(plot_width=260, plot_height=300, title="BEST(4)")
p34.image_url(url=["static/image/wc4.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p34.xaxis.visible = None
p34.yaxis.visible = None

x=df['Sentiment'].value_counts()
data = pd.Series(x).reset_index(name='Count').rename(columns={'index':'Sentiment'})
lp= (data['Count'] /data['Count'].sum() * 100).round(2)
lp=lp.tolist()
sp=[]
for i in range(5):
    v=str(lp[i])
    v=v+'%'
    sp.append(v)
data['percent']=sp
data.sort_values(by=['Sentiment'], inplace=True,ascending=True)
data1=data
data1.sort_values(by=['Count'], inplace=True,ascending=False)
nn=data1['Sentiment'].tolist()
mm=data1['percent'].tolist()
dd=data1['Count'].tolist()
source = ColumnDataSource(dict(
    x=[2, 4, 2.7, 3.1, 3.6],
    y=[1.5, 1.3, 0.9, 2.1, 1.8],
    color=Category10[5],
    yy=['AVERAGE','GOOD','BAD','BEST','WORST'],
    label=nn,
    radius= [0.9,0.75,0.6,0.45,0.33],
    v=nn,
    c=mm,
    names=nn,
    b=dd
))
p = figure(x_range=(0, 7), y_range=(0.5, 3), plot_height=400,plot_width=425, tools="zoom_in,zoom_out,save,reset",toolbar_location='right')
p.circle( x='x', y='y', radius='radius', color='color', legend_field='yy', source=source,fill_alpha=0.7)
labels1 = LabelSet(x='x', y='y', text='names',x_offset=0,y_offset=-4,
               source=source, render_mode='canvas',text_font_size="12px",text_color="black",text_font_style='bold')
labels2 = LabelSet(x='x', y='y', text='c',x_offset=-10,y_offset=-15,
               source=source, render_mode='canvas',text_font_size="10px",text_color="white")
hover=HoverTool()
hover.tooltips="""
<div>
<div><strong>Sentiment  : </strong>@v</div>
<div><strong>Percentage : </strong>@c</div>
<div><strong>Count : </strong>@b</div>
</div>
"""
p.add_tools(hover)
p.axis.visible = None
p.xgrid.visible = False
p.ygrid.visible = False
p.title.text = "Percentage of Sentiment"
p.title.align = "left"
p.title.text_font_size = "15px"
output_notebook()
output_file("Senti.html")
p.add_layout(labels1)
p.add_layout(labels2)


x=df['Sentiment'].value_counts()
data = pd.Series(x).reset_index(name='Count').rename(columns={'index':'Sentiment'})
data['angle'] = data['Count']/data['Count'].sum() * 2*pi
data['percent'] = data['Count'] /data['Count'].sum() * 100
data['color'] = Category20[len(x)]
p5 = figure(plot_height=400,plot_width=425, title="Percentage of each sentiments",
           tools="hover", tooltips="@Sentiment: @percent{0.2f} %", x_range=(-0.5, 1.0))
p5.wedge(x=0.3, y=1, radius=0.4,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', source=data)
p5.axis.axis_label=None
p5.axis.visible=False
p5.grid.grid_line_color = None




df0 =new_explode[(new_explode.Sentiment==0)]
df0.sort_values(by=['Phrase'], inplace=True,ascending=False)
x=pd.Series(' '.join(df0['Phrase']).split()).value_counts()
data0 = x.reset_index(name='Count').rename(columns={'index':'clean_Phrase'})
df0=data0.head(30) 
#df0.sort_values(by=['Count'], inplace=True,ascending=False)

df1 =new_explode[(new_explode.Sentiment==1)]
df1.sort_values(by=['Phrase'], inplace=True,ascending=False)
x=pd.Series(' '.join(df1['Phrase']).split()).value_counts()
data1 = x.reset_index(name='Count').rename(columns={'index':'clean_Phrase'})
df1=data1.head(30) 
#df0.sort_values(by=['Count'], inplace=True,ascending=False)

df2 =new_explode[(new_explode.Sentiment==2)]
df2.sort_values(by=['Phrase'], inplace=True,ascending=False)
x=pd.Series(' '.join(df2['Phrase']).split()).value_counts()
data2 = x.reset_index(name='Count').rename(columns={'index':'clean_Phrase'})
df2=data2.head(30) 
#df0.sort_values(by=['Count'], inplace=True,ascending=False)

df3 =new_explode[(new_explode.Sentiment==3)]
df3.sort_values(by=['Phrase'], inplace=True,ascending=False)
x=pd.Series(' '.join(df3['Phrase']).split()).value_counts()
data3 = x.reset_index(name='Count').rename(columns={'index':'clean_Phrase'})
df3=data3.head(30) 
#df0.sort_values(by=['Count'], inplace=True,ascending=False)

df4 =new_explode[(new_explode.Sentiment==4)]
df4.sort_values(by=['Phrase'], inplace=True,ascending=False)
x=pd.Series(' '.join(df4['Phrase']).split()).value_counts()
data4 = x.reset_index(name='Count').rename(columns={'index':'clean_Phrase'})
df4=data4.head(30) 
#df0.sort_values(by=['Count'], inplace=True,ascending=False)


columns = [
    TableColumn(field="clean_Phrase", title='CleanPhrase'),
    TableColumn(field="Count", title='Count')
    ]

def changeArea(attr, old, new):
    a=slider.value
    b=slider1.value
    if(a==0):
        source.data = df0.head(b)
        p41.image_url(url=["static/image/aa.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        p42.image_url(url=["static/image/Sentiment_test_cp0.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
    if(a==1):
        source.data = df1.head(b)
        p41.image_url(url=["static/image/bb.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        p42.image_url(url=["static/image/Sentiment_test_cp1.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
    if(a==2):
        source.data = df2.head(b)
        p41.image_url(url=["static/image/cc.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        p42.image_url(url=["static/image/Sentiment_test_cp2.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
    if(a==3):
        source.data = df3.head(b)
        p41.image_url(url=["static/image/dd.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        p42.image_url(url=["static/image/Sentiment_test_cp3.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
    if(a==4):
        source.data = df4.head(b)
        p41.image_url(url=["static/image/ee.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        p42.image_url(url=["static/image/Sentiment_test_cp4.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
        
p41 = figure(plot_width=300, plot_height=300, title="SENTIMENT_emoji")
p42 = figure(plot_width=300, plot_height=300, title="SENTIMENT_wordcloud")

p6= DataTable(source=source, columns=columns, width=700, height=400)

p41.image_url(url=["static/image/aa.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")
p42.image_url(url=["static/image/Sentiment0.png"], x=0, y=0, w=0.8, h=0.8,anchor="bottom_left")

slider = Slider(start=0, end=4, value=0, step=1, title="Sentiment Slider",bar_color="green")
slider.on_change('value', changeArea)
slider1 = Slider(start=5, end=30, value=5, step=5, title="Top Slider",bar_color="red")
slider1.on_change('value', changeArea)



p41.xgrid.visible = False
p41.ygrid.visible = False
p41.xaxis.visible = None
p41.yaxis.visible = None

p42.xgrid.visible = False
p42.ygrid.visible = False
p42.xaxis.visible = None
p42.yaxis.visible = None

pre=Div(text=""" <div><h3><strong><center> Sentimental Analysis</center></strong><h3></div>""",
        align='center',style={'color':'DarkBlue','font-size':'40px','column-fill':'auto'})
col1 = row(p2,p4,p)
pre0=Div(text=""" <div><h3><strong><center><br>Data Insights</br></center></strong><h3></div>""",
        align='start',style={'color':'DarkBlue','font-size':'20px','font-family':'Helvetica','column-fill':'auto'})
pre_br = Div(text=""" </br></br>""")
col2=row(p30,p31,p32,p33,p34)
pre1=Div(text=""" <div><h3><strong><center><br>Predictions</br></center></strong><h3></div>""",
        align='start',style={'color':'DarkBlue','font-size':'20px','font-family':'Helvetica','column-fill':'auto'})
col3 = row(slider,slider1)
col4 = row(p6,p41,p42)
from bokeh.io import curdoc
#layout = column(widgetbox(slider,slider1), p6)
layout=column(pre,pre0,col1,pre_br,col2,pre1,col3,col4)
curdoc().add_root(layout)
#show(layout)








































