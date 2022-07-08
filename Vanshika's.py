from selenium import webdriver
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from time import sleep
from selenium.webdriver.common.keys import Keys
from sklearn import preprocessing
from bs4 import BeautifulSoup

PATH = 'G:\chromedriver.exe'
option = webdriver.ChromeOptions() 

option.add_argument("--disable-infobars")
option.add_argument("--disable-popup-blocking")
option.add_argument("start-maximized")

option.add_argument("--disable-extensions")

option.add_experimental_option("prefs", 
{"profile.default_content_setting_values.notifications": 2
 })
driver = webdriver.Chrome(chrome_options=option, executable_path=PATH)

url = 'https://www.magicbricks.com/property-for-sale/residential-real-estate?&proptype=Multistorey-Apartment,Builder-Floor-Apartment,Penthouse,Studio-Apartment,Residential-House,Villa&cityName=Noida'
driver.get(url)

# Scrolling the website till the end
height = 0
while True:
    #driver.execute_script("window.scroll(0, 0);")
    #sleep(5)
    n_height = driver.execute_script("return document.body.scrollHeight")
    #if height == n_height and not(driver.find_element_by_class('pageLoader')):
    if height == n_height:
        break
    driver.find_element_by_tag_name('body').send_keys(Keys.END)
    sleep(10)
    height = n_height

html = driver.page_source
soup = BeautifulSoup(html)
con = soup.find_all('div',class_ = 'mb-srp__card__info')
container2 = soup.find_all('div',class_ = 'mb-srp__card__estimate')

project = []
location = []
bhk = []
total_floors = []
furnishing = []
bathroom = []
carpet_area = []
super_area = []
price = []
rate = []


for c in con:
    p = c.h2.text
    s_area=0
    flo = 0
    c_area = 0
    b = p[0]
    fur = 0 
    bath = 0
    if b == ">":
        b = p[0:4]
    pro = ""
    loc = p.split("in ",1)[1]
    if c.find('div','mb-srp__card__society') is not None:
        g = c.find('div','mb-srp__card__society').a
        pro = g.text
    
    suga = c.find_all('div',class_='mb-srp__card__summary--label')
    su = c.find_all('div',class_ = 'mb-srp__card__summary--value')
    i = 0
    for x in suga:
        if x.text == 'Super Area':
            a = su[i].text
            a = a.replace(',','')
            s_area = a.split(' ',1)[0]
            #cutting_last(sup)
        if x.text == 'Floor':
            q = su[i].text
            try:
                flo = q.split('of ',1)[1]
            except IndexError:
                flo = q
                #cutting_f(fl)
        if x.text == 'Carpet Area':
            a = su[i].text
            a = a.replace(',','')
            c_area = a.split(' ',1)[0]
            
        if x.text == 'Furnishing':
            fur = su[i].text
            
        if x.text == 'Bathroom':
            bath = su[i].text

        i = i+1
            
    project.append(pro)
    bhk.append(b)
    location.append(loc)
    total_floors.append(flo)
    furnishing.append(str(fur))
    bathroom.append(bath)
    super_area.append(float(s_area))
    carpet_area.append(float(c_area))


    for e in container2:
    pri = 0
    r = 0
    if e.find('div',class_='mb-srp__card__price--amount') is not None:
        p = e.find('div',class_='mb-srp__card__price--amount')
        p1 = p.text[1:]
        p2 = p1.split(" ", 1)[0]
        p3 = p1.split(" ",1)[1]
        pri = p2.replace(',', '')
        pri = float(pri)
        if p3 == "Lac":
            pri *= 100000
        if p3 == "Cr":
            pri *= 10000000
        
        
    if e.find('div', class_ = 'mb-srp__card__price--size') is not None:
        ro = e.find('div', class_ = 'mb-srp__card__price--size')
        r1 = ro.text[1:]
        r2 = r1.split(" ",1)[0]
        r = r2.replace(',', '')
        
    rate.append(float(r))
    price.append(float(pri))
    

driver.quit()


import pandas as pd
dic = {
    'project' : project,
    'location' : location,
    'bhk' : bhk,
    'total_floors' : total_floors,
    'furnishing' : furnishing,
    'bathroom' : bathroom,
    'carpet area' : carpet_area,
    'super_area' : super_area,
    'price' : price,
    'rate' : rate
}
df = pd.DataFrame(dic)

df.to_csv('details.csv', index = False)

df.head()

df.dropna(subset=['project','bhk', 'price', 'total_floors','location','rate'], axis=0, inplace=True)

df['bhk'] = df['bhk'].str.replace('> 10','10')
df['bhk'] = df['bhk'].str.replace(' ','')
df['bhk'] = df['bhk'].replace(r'^\s*$', 0, regex=True)
df['bhk'] = df['bhk'].astype(int)
df['total_floors'] = df['total_floors'].replace('Ground', "1", regex=True)
df['total_floors'] = df['total_floors'].astype(int)


df['bathroom'] = df['bathroom'].replace('> 10', '11', regex=True)
df['bathroom'] = df['bathroom'].astype(int)
df['project'] = df['project'].astype(str)
df['location'] = df['location'].astype(str)
df['furnishing'] = df['furnishing'].astype(str)

df.drop(df.index[df['bhk'] == 0], inplace=True)
df.drop(df.index[df['total_floors'] == 0], inplace=True)
df.drop(df.index[df['rate'] == 0], inplace=True)

#Removing spaces from starting from location
df['location'] = df['location'].str.strip()
df['project'] = df['project'].str.strip()

df.drop(df.index[df['bhk'] == ">"], inplace=True)
df["bhk"] = pd.to_numeric(df["bhk"])
df["bhk"] = df["bhk"].astype(int)

df = df.drop_duplicates()

def g(row):
    if (row['carpet area']) != 0:
    #if row['Raw Carpet Area'] is notnull():
        val = row['carpet area']
    elif row['super_area'] != 0:
    #elif row['Raw Super Area'] is notnull():
        val = row['super_area']/1.5
    else:
        val = None
    return val
df['Carpet Area'] = df.apply(g, axis=1)


x_cols = ['project','location','total_floors','bhk','Carpet Area']
y_cols = ['rate']


new_data = df.copy()

X = new_data[x_cols]
Y = new_data[y_cols]

from sklearn.preprocessing import LabelEncoder,Normalizer,MinMaxScaler,StandardScaler
LE_project=LabelEncoder()

X['project_code'] = LE_project.fit_transform(X['project'])
X['location_code'] = LE_project.fit_transform(X['location'])

new_data['project_code'] = X['project_code']
new_data['location_code'] = X['location_code']
X.drop(['project','location'],axis=1,inplace=True)

#Training different models

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier  
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101) 

# Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
predictions = lin_reg.predict(X_test)

# Graident Boosting
from sklearn.ensemble import GradientBoostingRegressor
gradient_boosting = GradientBoostingRegressor(n_estimators=500, max_depth=8, learning_rate=0.05, random_state=42)
# print(x_train,y_train)
print(type(X_train),type(y_train))
gradient_boosting.fit(X_train, y_train)
gred = gradient_boosting.predict(X_test)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
random_forest = RandomForestRegressor(n_estimators=50, random_state=0)
random_forest.fit(X_train, y_train)
ran = random_forest.predict(X_test)

# Random Forest Regressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
rf_random = RandomForestRegressor()
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
random_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf' : min_samples_leaf}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, scoring = 'neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)
random_forest = rf_random.predict(X_test)

t = X_test
t['Actual Rate'] = y_test
t['Linear Regression'] = predictions
t['Random_forest_Regressor'] = random_forest
t['Random Forest'] = ran
t['Gradient Boosting'] = gred
t.to_csv('test_data.csv', index=False)
