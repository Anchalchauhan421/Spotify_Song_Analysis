import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sp_track= pd.read_csv(r'C:\Users\ANCHAL\tracks.csv',encoding='latin-1')
sp_track

sp_track.head()

#find the null values
pd.isnull(sp_track).sum()

sp_track.info()

#10 lesat popular song in ds
sorted_df = sp_track.sort_values('popularity',ascending=True).head(10)
sorted_df


sp_track.describe().transpose()

#10 most popular songs >90 
most_popular=sp_track.query('popularity>90',inplace=False).sort_values('popularity',ascending=False)
most_popular[:10] #slicing operator for choiceing top 10 song . 


sp_track.set_index("release_date",inplace=True)
sp_track.index = pd.to_datetime(sp_track.index)
sp_track.head()



sp_track[["artists"]].iloc[18] #check index number in data set 

sp_track["duration"]=sp_track["duration_ms"].apply(lambda x: round(x/1000))
sp_track.drop("duration_ms",inplace=True,axis=1)

sp_track.duration.head()

corr_df=sp_track.drop(["key","mode","explicit"],axis=1).corr(method="pearson")
plt.figure(figsize=(16,8))
heatmap=sns.heatmap(corr_df,annot=True,fmt=".1g",vmin=-1,vmax=1,center=0,cmap="inferno",linewidths=1,linecolor="Black")
heatmap.set_title("Correlation Heatmap Between Variable")
heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation=90)

sample_df=sp_track.sample(int(0.004*len(sp_track)))

print(len(sample_df))

plt.figure(figsize=(10,6))
sns.regplot(data=sample_df,y="loudness",x="energy",color="c").set(title="Loudness vs Energy Correlation")


plt.figure(figsize=(14,8))
sns.regplot(data=sample_df,y="popularity",x="acousticness",color="b").set(title="Popularity vs Acousticness Correlation")


sp_track['dates']=sp_track.index.get_level_values('release_date')
sp_track.dates=pd.to_datetime(sp_track.dates)
years=sp_track.dates.dt.year

#pip install---user seaborn=0.11.0


sns.displot(years,discrete=True,aspect=2,height=5,kind="hist").set(title="Number of songs per year")

total_dr=sp_track.duration
fig_dims=(18,7)
fig,ax=plt.subplots(figsize=fig_dims)
fid=sns.barplot(x=years,y=total_dr,ax=ax,errwidth=False).set(title="Year vs Duration ")
plt.xticks(rotation=90)

df_feat=pd.read_csv(r"E:\Dataset\SpotifyFeatures.csv")
df_feat

df_feat.head()

plt.title("Duration of the Songs in Different Genres")
sns.color_palette("rocket",as_cmap=True)
sns.barplot(y='genre',x='duration_ms',data=df_feat)
plt.xlabel("Duration in milli seconds")
plt.ylabel("Genres")

sns.set_style(style='darkgrid')#for background
plt.figure(figsize=(10,5))
famous=df_feat.sort_values("popularity",ascending=False).head(10)
sns.barplot(y='genre',x='popularity',data=famous).set(title="Top 5 Genres by popularity")









