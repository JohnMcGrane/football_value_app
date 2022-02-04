import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import streamlit as st
import pandas as pd
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

def results_visualization(overvaluedf,undervaluedf):
	plt.style.use('dark_background')
	fig, ax = plt.subplots(figsize=(13,10))

	y_pos = np.arange(10)
	red_cmap = plt.get_cmap('Reds')
	green_cmap = plt.get_cmap('Greens')

	red_rescale = lambda y: y / (np.min(y))
	green_rescale = lambda y: y / (np.max(y))

	ax.barh(y_pos, overvaluedf['Value over MV'], align='center', color=red_cmap(red_rescale(overvaluedf['Value over MV'])))
	ax.barh(y_pos, undervaluedf['Value over MV'][::-1], align='center', color=green_cmap(green_rescale(undervaluedf['Value over MV'][::-1])))

	ax.set_yticks(y_pos)
	ax.tick_params(axis ='y')
	ax.set_yticklabels(overvaluedf['Player'],fontsize=14)

	plt.tick_params(axis='x', which='both', bottom=False,top=False,labelbottom=False)
	ax2 = ax.twinx()
	ax2.set_yticks(np.linspace(0.08,0.925,10))
	ax2.tick_params(axis ='y')
	ax2.set_yticklabels(undervaluedf['Player'][::-1],fontsize=14)

	for count, value in enumerate(overvaluedf['Value 10M']):
		height = np.linspace(0.07,0.905,10)
		plt.text(x = -22000000,y=height[count],s=value,fontsize=17,weight = 'bold')
		plt.text(x = 1000000,y=height[count],s=undervaluedf['Value 10M'][::-1].iloc[count],fontsize=17,weight = 'bold')

	ax.set_title(f'Predicted Value Compared to Market',fontsize=20,weight = 'bold')
	plt.grid(axis='x',color = 'black',alpha = 0.2)
	return fig

def model_selector(optn):
	if optn == "GradientBoostingRegressor":
		regressor = GradientBoostingRegressor(random_state = 3,max_features='sqrt')
	elif optn == "RandomForestRegressor":
		regressor = RandomForestRegressor(n_estimators=50,random_state = 2)
	elif optn == "LinearRegression":
		regressor = LinearRegression()
	return regressor

def df_format(data1,valuediff1):
	data1['Value over MV'] = valuediff1
	valuedf = data1.sort_values(by=['Value over MV'],ascending = False)

	valuedf['Value'] = valuedf['Value over MV'].map('${:,.2f}'.format)
	valuedf['Value 10M'] = (valuedf['Value over MV']/1000000).map('${:,.2f}m'.format)

	undervaluedf = valuedf[['Player','Value over MV','Value 10M','Value']][:10]
	overvaluedf = valuedf[['Player','Value over MV','Value 10M','Value']][-10:]
	return overvaluedf,undervaluedf


@st.cache(persist=True,suppress_st_warning=True)
def player_page_urls():
	column_names = ["Link_"]
	df = pd.DataFrame(columns = column_names)
	headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}
	url = "https://www.transfermarkt.us/spieler-statistik/wertvollstespieler/marktwertetop/plus/ajax/yw1/ausrichtung/Sturm/spielerposition_id/alle/altersklasse/alle/jahrgang/0/land_id/0/kontinent_id/0/yt0/Show/0//page/"

	for page in range(1,21):
		newurl = url + str(page)
		request = requests.get(newurl,  headers=headers)
		soup = BeautifulSoup(request.content, 'html.parser')
		players = soup.find_all("tr", class_=["odd","even"])
		for count, player in enumerate(players):
			# Player Link
			link = player.find(class_="hauptlink")
			link = link.find('a', href=True)['href']
			dflinks = pd.DataFrame([[link]], columns = column_names)
			df = df.append(dflinks, ignore_index=True)
	return df

def player_name(sp):
    try:
	    nameset = sp.find_all(class_="dataName")
	    name = nameset[0].find_all('h1')[0].text.strip()
	except:
	    name = np.nan
	return name

def season_domestic_games(sp):
    try:
        goal_table1 = sp.find_all(attrs={"data-viewport": "Leistungsdaten_Saison"})
        goal_table = goal_table1[0].find_all('td',class_="zentriert")
        games_played = goal_table[:5][0].find(text=True)
        games_played = zero_stats(games_played)
    except:
        games_played = np.nan
	return games_played

def season_domestic_goals(sp):
    try:
        goal_table1 = sp.find_all(attrs={"data-viewport": "Leistungsdaten_Saison"})
        goal_table = goal_table1[0].find_all('td',class_="zentriert")
        goals = goal_table[:5][1].find(text=True)
        goals = zero_stats(goals)
    except:
        goals = np.nan
	return goals

def season_domestic_assists(sp):
    try:
        goal_table1 = sp.find_all(attrs={"data-viewport": "Leistungsdaten_Saison"})
        goal_table = goal_table1[0].find_all('td',class_="zentriert")
        assists = goal_table[:5][2].find(text=True)
        assists = zero_stats(assists)
    except:
        assists = np.nan
	return assists

def season_domestic_mpg(sp): # minutes per goal
    try:
        goal_table1 = sp.find_all(attrs={"data-viewport": "Leistungsdaten_Saison"})
        goal_table = goal_table1[0].find_all('td',class_="zentriert")
        mpg = goal_table[:5][3].find(text=True)
        mpg = re.sub("\.", "", mpg)
        mpg = zero_stats(mpg)
    except:
        mpg = np.nan
	return mpg

def season_domestic_total_mins(sp): # total minutes played
	goal_table1 = sp.find_all(attrs={"data-viewport": "Leistungsdaten_Saison"})
	goal_table = goal_table1[0].find_all('td',class_="zentriert")
	total_mins = goal_table[:5][4].find(text=True) # Total Minutes
	total_mins = re.sub("\.", "", total_mins)
	total_mins = zero_stats(total_mins)
	return total_mins

def current_market_value(sp):
	value1 = sp.find_all(class_="marktwertentwicklung")[0] # CURRENT VALUE
	valueString = value1.find(class_="zeile-oben").find_all(class_="right-td")[0].find_all(text=True)
	if len(valueString) > 1:
		valueString = valueString[1]
	else:
		valueString = valueString[0]
	wage_multiplier = multiplier(valueString)
	regex_pattern = pattern(valueString)
	p = re.compile(regex_pattern)
	current_val = p.findall(valueString)[0]
	current_val = floatable_string(current_val, wage_multiplier)
	return current_val

def market_value_at_purchase(sp):
	value = sp.find_all(class_="zelle-mw") # PURCHASE VALUE
	valueString = value[1].find_all(text=True)[0]
	wage_multiplier = multiplier(valueString)
	regex_pattern = pattern(valueString)
	p = re.compile(regex_pattern)
	valueString = p.findall(valueString)[0]
	purchase_val = zero_stats(valueString)*wage_multiplier
	return purchase_val

def purchase_fee(sp):
	value = sp.find_all(class_="zelle-abloese") # PURCHASE FEE
	valueString = value[1].find_all(text=True)[0]
	wage_multiplier = multiplier(valueString)
	regex_pattern = pattern(valueString)
	p = re.compile(regex_pattern)
	valueString = p.findall(valueString)[0]
	purchase_fee = floatable_string(valueString, wage_multiplier)
	return purchase_fee

def player_age(sp):
	age = sp.find_all(class_="large-6 large-pull-6 small-12 columns spielerdatenundfakten")
	datastring = age[0].text.strip()
	age = float(datastring.split("Age",1)[1][2:4])
	return age

def player_height(sp):
	ht = sp.find_all(class_="large-6 large-pull-6 small-12 columns spielerdatenundfakten")
	datastring = ht[0].text.strip()
	height = datastring.split("Height",1)[1][2:6]
	height = float(re.sub("\,", ".", height))
	return height

def league_tier(sp):
	value = sp.find_all(class_="dataZusatzDaten")[0]
	value1 = value.find_all(class_="dataValue")[0]
	tier = value1.find_all(text=True)[1].strip() # get league tier and strip whitespace
	return tier

def league_country(sp):
	value = sp.find_all(class_="dataZusatzDaten")[0]
	value1 = value.find_all(class_="dataValue")[0]
	country = value1.find_all("img")[0].get('title') # Get League Country
	return country

def club_team(sp):
	value = sp.find_all(class_="dataZusatzDaten")[0]
	value1 = value.find_all(class_="hauptpunkt")[0]
	team = value1.find_all(text=True)[0]
	return team

def player_number(sp):
	value = sp.find_all(class_="dataHeader dataExtended")[0]
	try:
		value1 = value.find_all(class_="dataRN")[0]
		number = value1.find_all(text=True)[0]
		number = float(re.sub("\#", "", number))
	except:
		number = 0
	return number

def player_nationality(sp):
	nation = sp.find_all(class_="dataHeader dataExtended")[0]
	try:
		nation1 = nation.find_all(class_="flaggenrahmen flagge")[0]
	except:
		nation1 = nation.find_all(class_="flaggenrahmen")[0]
	nationality = nation1.get('title')
	return nationality

# The string "$176.00m" is returned as the string "176.00"
def pattern(string):
	if "Th" in string:
		pattern = "(?<=\$)(.*?)(?=Th)"
	elif "m" in string:
		pattern = "(?<=\$)(.*?)(?=m)" # pattern for stripping the wage number
	else:
		pattern = ".*"
	return pattern

# Set the multiplier according to value in the millions or thousands
def multiplier(string):
	multiplier = 0
	if "Th" in string:
		multiplier = 1000
	elif "m" in string:
		multiplier = 1000000
	return multiplier

# If a stat string is listed as '-' return this as zero, otherwise convert string to float
def zero_stats(string):
	if string == "-":
		return_val = 0
	else:
		return_val = float(string)
	return return_val

# Convert string to float (if possible) and multiply, otherwise maintain string
def floatable_string(string, multiplier):
	try:
		return_val = float(string)*multiplier
	except:
		return_val = string
	return return_val


def format_and_merge_data(df1):
	scrape_df = df1.loc[df1['Games Played'] <= 40].reset_index()
	surname = []
	firstInitial = []
	for i in scrape_df['Player']:
		surname.append(i.split()[-1])
		firstInitial.append(i[0])
	scrape_df['Surname'] = surname
	scrape_df['Initial'] = firstInitial

	fifadf = pd.read_csv("FIFA22_official_data.csv")
	lastName = []
	firstInitial2 = []
	for i in fifadf['Name']:
		lastName.append(i.split()[-1])
		firstInitial2.append(i[0])
	fifadf['Surname'] = lastName
	fifadf['Initial'] = firstInitial2
	fifadf = fifadf.drop(columns = ['Age','Height'])

	merge1 = scrape_df.merge(fifadf, how = "left", on = ['Surname', 'Initial'])
	#print(merge1.shape)
	merge2 = merge1.drop_duplicates(subset=["Player"], keep='first')
	#print(merge2.shape)
	merge3 = merge2.dropna(subset=['Name'])
	merge4 = merge3.reset_index(drop=True)
	merged = merge4.drop(columns=['Jersey Number_y', 'Initial', 'ID', 'Surname', 'Photo',
							'Flag', 'Club', 'Club Logo', 'Value' , 'Wage', 'Special', 'Work Rate', 'Position',
							'Real Face', 'Joined', 'Loaned From', 'Release Clause', 'GKReflexes', 'GKPositioning',
							'GKKicking', 'GKHandling','GKDiving', 'Weight', 'Contract Valid Until', 'Marking',
							'index', 'Jersey Number_x', 'Club Country', 'Tier', 'Team',
							'Nationality_y','Nationality_x', 'Preferred Foot','Name','Body Type','Best Position'])
	return merged






