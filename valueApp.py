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

# TITLE
# -----------------------------------------------------------
st.title("Undervalued Assets")
st.write("## Finding Untapped Potential on the Pitch")
# -----------------------------------------------------------

#@st.cache(persist=True,suppress_st_warning=True)
#@st.experimental_memo(suppress_st_warning=True)
def scrape_data(df):
    headers = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36"}
    column_names = ["Player", "Games Played", "Goals", "Assists", "Minutes per Goal", "Total Minutes Played", 
                    "Current Market Value", "Market Value at Purchase", "Age", "Height", "Club Country", 
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
        mpg = season_domestic_mpg(soup)
        minutes = season_domestic_total_mins(soup)
        MV = current_market_value(soup)
        prev_MV = market_value_at_purchase(soup)
        #fee = purchase_fee(soup)
        age = player_age(soup)
        height = player_height(soup)
        name = player_name(soup)
        country = league_country(soup)
        tier = league_tier(soup)
        team = club_team(soup)
        number = player_number(soup)
        nationality = player_nationality(soup)
        #st.write(ct)
        ct+=1
        load_bar.progress(float(ct/len(df['Link_'])))
        df2 = pd.DataFrame([[name, games, goals, assists, mpg, minutes, MV, prev_MV, age, 
                             height, country, tier, team, number, nationality]], columns = column_names)
        df1 = df1.append(df2, ignore_index=True)
    return df1


# DROPDOWN MENU SCRAPE DATA OPTION
# -----------------------------------------------------------
opt1 = st.selectbox('Scrape Transfermarkt for current data? (This will take several minutes. Not recommended)',
   ('Yes', 'No'),index=1)
# -----------------------------------------------------------
if opt1 == "Yes":
    linkdf = player_page_urls()
    #st.write(linkdf)
    scrp_Data = scrape_data(linkdf)
    data = format_and_merge_data(scrp_Data)
    st.write(data)
else:
    data = pd.read_csv("fifavaluedf.csv")

# linkdf = player_page_urls()
# scrp_Data = scrape_data(linkdf)
# st.write(scrp_Data)

#st.write(new)

# DROPDOWN MENU MODEL SELECTION
# -----------------------------------------------------------
option = st.selectbox('Model Selection',
    ('RandomForestRegressor', 'GradientBoostingRegressor', 
        'LinearRegression'),index=2)
# -----------------------------------------------------------

regressor = model_selector(option) # from app_utils.py


#data = pd.read_csv("fifavaluedf.csv")

@st.cache(persist=True,suppress_st_warning=True)
def loo_cv(data,regressor,optn):
    Y = data['Current Market Value']
    if optn == "Yes":
        X = data.drop(columns=['Current Market Value','Player'])
    else:
        X = data.drop(columns=['Current Market Value','Player','Unnamed: 0'])
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

r2score = r2_score(ytestlist, ypredlist)
st.write("## Model Performance")
st.write("Coefficient of Determination: ", r2score)

format_results =  df_format(data,value_diff)
overvaluedf = format_results[0]
undervaluedf = format_results[1]
st.write("## Results")

# DROPDOWN MENU RESULTS OPTION
# -----------------------------------------------------------
display = st.selectbox('Select Results Display:',
   ('Table', 'Bar Graph'),index=1)
# -----------------------------------------------------------



# RESULTS DISPLAY
# -----------------------------------------------------------
if display == 'Bar Graph':
    figure = results_visualization(overvaluedf,undervaluedf) # from app_utils.py
    st.pyplot(figure)
if display == 'Table':
    col1, col2 = st.columns(2)

    col1.header("Most Overvalued")
    col1.write(overvaluedf[::-1][['Player', 'Value']])
    #col1.write(valuedf[['Player','Value over MV']][-10:].iloc[::-1])

    col2.header("Most Undervalued")
    col2.write(undervaluedf[['Player', 'Value']])
    #col2.write(valuedf[['Player','Value over MV']][:10])
# -----------------------------------------------------------



