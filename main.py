from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import seaborn as sns

option = webdriver.ChromeOptions()
option.add_argument('headless')
option.add_argument('disable-gpu')
option.add_argument('log-level=3')
driver = webdriver.Chrome(options=option, service=Service(ChromeDriverManager().install()))

start = time.time()
print('Collecting Data...')

driver.get('https://www.koreahtml5.kr/front/stats/browser/browserUseStats.do')

# period when chrome started to record
Select(driver.find_element(
    By.XPATH, '//*[@id="startMonth"]')).select_by_visible_text('2009-08')
Select(driver.find_element(By.XPATH, '//*[@id="endMonth"]')).select_by_index(0)
Select(driver.find_element(
    By.XPATH, '//*[@id="countryNo"]')).select_by_visible_text('worldwide')

driver.find_element(By.XPATH, '//*[@id="searchForm"]/fieldset/button').click()
driver.find_element(
    By.CLASS_NAME, 'highcharts-a11y-proxy-button.highcharts-no-tooltip').click()
driver.find_element(By.CLASS_NAME,
                    'highcharts-contextmenu').find_element(By.XPATH, '//ul/li[9]').click()

print("Getting Categories")
category = [str(_.text) for _ in tqdm(driver.find_elements(
    By.XPATH, '//*[@id="highcharts-data-table-0"]/thead/tr/th[position() > 1]'))]
print("Getting Periods")
period = [str(_.text) for _ in tqdm(driver.find_elements(
    By.XPATH, '//*[@id="highcharts-data-table-0"]/tbody/tr/th'))]
print("Getting Data Labels")
contents = [float(_.text) for _ in tqdm(driver.find_elements(
    By.XPATH, '//*[@id="highcharts-data-table-0"]/tbody/tr/td'))]
driver.quit()

runtime = time.time() - start
print(f'Done. Elapsed Time = {runtime:.2f}s')

contentsList = []
for _ in range(0, len(contents), len(category)):
    contentsList.append(contents[_:_ + len(category)])

dataTable = pd.DataFrame(contentsList, columns=category)
dataTable['Period'] = period

X = dataTable[['Period']].astype('float')
Y = dataTable[['Chrome']]  # Any Browser you wanna predict

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3, random_state=10)

poly = PolynomialFeatures(degree=2)
X_Train_Poly = poly.fit_transform(X_Train)
pr = LinearRegression()
pr.fit(X_Train_Poly, Y_Train)
X_Test_Poly = poly.fit_transform(X_Test)
y_hat_test = pr.predict(X_Test_Poly)

X_poly = poly.fit_transform(X)
y_hat = pr.predict(X_poly)

plt.figure(figsize=(10, 5))
ax1 = sns.distplot(Y, hist=False, label='Y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax=ax1)
plt.show()
plt.close()

exit(0)
