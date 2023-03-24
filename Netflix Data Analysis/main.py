import numpy as np
import pandas as pd
import plotly.express as px
from textblob import TextBlob

data = pd.read_csv('D:/Kuljeet/Projects/Data-Analysis-Projects\DataSets/Netflix Data/netflix_titles.csv')
#
#print(data.shape)
#print(data.columns)
#
#z = data.groupby(['rating']).size().reset_index(name='counts')
#pieChart = px.pie(z, values='counts', names='rating', 
#                  title='Distribution of Content Ratings on Netflix',
#                  color_discrete_sequence=px.colors.qualitative.Set3)
#pieChart.show()

data['director']=data['director'].fillna('No Director Specified')
filtered_directors=pd.DataFrame()
filtered_directors=data['director'].str.split(',',expand=True).stack()
filtered_directors=filtered_directors.to_frame()
filtered_directors.columns=['Director']
directors=filtered_directors.groupby(['Director']).size().reset_index(name='Total Content')
directors=directors[directors.Director !='No Director Specified']
directors=directors.sort_values(by=['Total Content'],ascending=False)
directorsTop5=directors.head()
directorsTop5=directorsTop5.sort_values(by=['Total Content'])
fig1=px.bar(directorsTop5,x='Total Content',y='Director',title='Top 5 Directors on Netflix')
fig1.show()

df1=data[['type','release_year']]
df1=df1.rename(columns={"release_year": "Release Year"})
df2=df1.groupby(['Release Year','type']).size().reset_index(name='Total Content')
df2=df2[df2['Release Year']>=2010]
fig2 = px.line(df2, x="Release Year", y="Total Content", color='type',title='Trend of content produced over the years on Netflix')
fig2.show()

dfx=data[['release_year','description']]
dfx=dfx.rename(columns={'release_year':'Release Year'})
for index,row in dfx.iterrows():
    z=row['description']
    testimonial=TextBlob(z)
    p=testimonial.sentiment.polarity
    if p==0:
        sent='Neutral'
    elif p>0:
        sent='Positive'
    else:
        sent='Negative'
    dfx.loc[[index,2],'Sentiment']=sent


dfx=dfx.groupby(['Release Year','Sentiment']).size().reset_index(name='Total Content')

dfx=dfx[dfx['Release Year']>=2010]
fig3 = px.bar(dfx, x="Release Year", y="Total Content", color="Sentiment", title="Sentiment of content on Netflix")
fig3.show()
