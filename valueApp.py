import numpy as np
import pandas as pd
import streamlit as st
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from app_utils import *

# Introduction
# -----------------------------------------------------------
st.title("Undervalued Assets on the Pitch")
st.write("International football is an industry flush with cash. In addition to on-field performance, egos, narratives, expectations, and popularity all play a critical role in determining the market value of a player. The goal of this application is to build and compare several models that predict player value.")
st.write("## Data Information")
st.write("The data used in this analysis is scraped from the German footballing website *Transfermarkt*. For this analysis, data for the 500 most valuable forwards in world football were used. The default data was scraped on 1.15.22, but there is an option to update these data by running a scraping procedure through the app. For more features, statistics from the FIFA 22 videogame were merged. Due to the lack of a unique key to merge both datasets, the final dataset comprises only 400 players.")
st.write("## Data")
# ***********************************************************

# DATA SCRAPE OR IMPORT SECTION
# -----------------------------------------------------------
@st.cache(suppress_st_warning=True) # persist=True,
def scrape_data(df):
	headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}
	column_names = ["Player", "Games Played", "Goals", "Assists", "Age", "Height", "Minutes per Goal", "Total Minutes Played", 
					"Current Market Value", "Market Value at Purchase", "Club Country", 
					"Tier", "Team", "Jersey Number","Nationality"]
	df1 = pd.DataFrame(columns = column_names)
	ct = 0
	st.write("Scraping progress...")
	load_bar = st.progress(0)
	for link in df['Link_']:
		player_url = 'https://www.transfermarkt.us' + link
		request = requests.get(player_url, headers=headers)
		soup = BeautifulSoup(request.content, 'html.parser')

		games = season_domestic_games(soup)
		goals = season_domestic_goals(soup)
		assists = season_domestic_assists(soup)
		age = player_age(soup)
		height = player_height(soup)
		mpg = season_domestic_mpg(soup)
		minutes = season_domestic_total_mins(soup)
		MV = current_market_value(soup)
		prev_MV = market_value_at_purchase(soup)
		name = player_name(soup)
		country = league_country(soup)
		tier = league_tier(soup)
		team = club_team(soup)
		number = player_number(soup)
		nationality = player_nationality(soup)
		ct+=1
		load_bar.progress(float(ct/len(df['Link_'])))
		df2 = pd.DataFrame([[name, games, goals, assists, age, height, mpg, minutes, MV, prev_MV,  
							 country, tier, team, number, nationality]], columns = column_names)
		df1 = df1.append(df2, ignore_index=True)
	return df1

opt1 = st.selectbox('Scrape Transfermarkt for current data? (This will take several minutes. Not recommended)',
   ('Yes', 'No'),index=1)
if opt1 == "Yes":
	linkdf = player_page_urls()
	scrp_Data = scrape_data(linkdf)
	data = format_and_merge_data(scrp_Data)
	data.style.hide_index()
	st.write(data.head())
else:
	data = pd.read_csv("fifavaluedf.csv")
	data = data.drop(columns=['Unnamed: 0'])
	data = data.rename(columns={"Age_x": "Age", "Height_x": "Height"})
	st.write(data.head())
# ***********************************************************


# EXPLORE DATA SECTION 
# -----------------------------------------------------------
st.write("## Explore Data")
yvariable = st.selectbox('Select feature to view relation to market value',
   ((list(data.columns)[1:])),index=0)

def scatter_plot(x,y):
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(10,6))
	plt.scatter(x,y,color = 'firebrick')
	plt.ylabel(f'{y.name} ($)',fontsize=20)
	plt.xlabel(x.name,fontsize=20)
	ax.tick_params(axis='both', size= 14)
	return fig

fig1 = scatter_plot(data[yvariable],data['Current Market Value']) 
st.pyplot(fig1)
# ***********************************************************


# MODEL SELECT & RUN SECTION
# -----------------------------------------------------------
st.write("## Model Selection")
st.write("")
option = st.selectbox(label = 'Choose a model for the data',options=('RandomForestRegressor', 'GradientBoostingRegressor', 
		'LinearRegression'),index=2)


regressor = model_selector(option) # from app_utils.py

@st.cache(persist=True,suppress_st_warning=True)
def loo_cv(data,regressor,optn):
	Y = data['Current Market Value']
	#if optn == "Yes":
	X = data.drop(columns=['Current Market Value','Player'])
	#else:
	#	X = data.drop(columns=['Current Market Value','Player','Unnamed: 0'])
	loo = LeaveOneOut()
	loo.get_n_splits(X)
	X = np.array(X)
	X = StandardScaler().fit_transform(X)
	Y = np.array(Y)
	ix = 0
	ypredlist = np.zeros(len(X))
	ytestlist = np.zeros(len(X))
	load_bar = st.progress(0)
	for train_index, test_index in loo.split(X):
		Xtrain= X[train_index, :] 
		Xtest =  X[test_index, :]
		Ytrain, Ytest = Y[train_index], Y[test_index]
		model = regressor.fit(Xtrain, Ytrain)
		y_pred = model.predict(Xtest)
		ypredlist[ix] = y_pred
		ytestlist[ix] = Ytest
		#st.write(f'{ix}', end="")
		ix=ix+1
		load_bar.progress(float(ix/len(ypredlist)))
	return ypredlist,ytestlist

cv_results = loo_cv(data,regressor,opt1)
ypredlist = cv_results[0]
ytestlist = cv_results[1]
value_diff = ypredlist - ytestlist
# ***********************************************************


# MODEL PERFORMANCE SECTION
# -----------------------------------------------------------
r2score = r2_score(ytestlist, ypredlist)
r2score_formatted= ("{:.2f}%".format(r2score*100))
st.write("## Model Performance")
st.write("Coefficient of Determination: ", r2score)
st.write("Regression models can be evaluated based on their r2 score, or their coefficient of determination. The r2 score tells us what percent of the variation in the y variable can be explained by the variations in the x variables.")
st.write("In this case, ",r2score_formatted, "of a player's current market value can be explained by the independent variables in this model.")
# ***********************************************************


# RESULTS SECTION
# -----------------------------------------------------------
st.write("## Final Results")
display = st.selectbox('Select Results Display:',
   ('Table', 'Bar Graph'),index=1)
format_results =  df_format(data,value_diff)
overvaluedf = format_results[0]
undervaluedf = format_results[1]
if display == 'Bar Graph':
	figure = results_visualization(overvaluedf,undervaluedf) # from app_utils.py
	st.pyplot(figure)
if display == 'Table':
	col1, col2 = st.columns(2)
	col1.header("Most Overvalued")
	col1.write(overvaluedf[::-1][['Player', 'Value']].reset_index(drop=True))
	col2.header("Most Undervalued")
	col2.write(undervaluedf[['Player', 'Value']].reset_index(drop=True))
# ***********************************************************



