import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# ------------------ Q1: Books to Scrape ------------------
def scrape_books():
    base_url = "https://books.toscrape.com/catalogue/page-{}.html"
    books_data = []
    page = 1
    while True:
        url = base_url.format(page)
        response = requests.get(url)
        if response.status_code != 200:
            break
        soup = BeautifulSoup(response.text, "html.parser")
        articles = soup.find_all("article", class_="product_pod")
        if not articles:
            break
        for article in articles:
            title = article.h3.a["title"]
            price = article.find("p", class_="price_color").text.strip()
            availability = article.find("p", class_="instock availability").text.strip()
            star_rating = article.p["class"][1]
            books_data.append([title, price, availability, star_rating])
        page += 1

    df_books = pd.DataFrame(books_data, columns=["Title", "Price", "Availability", "Star Rating"])
    df_books.to_csv("books.csv", index=False)
    print("Books data saved to books.csv")

# ------------------ Q2: IMDB Top 250 ------------------
def scrape_imdb():
    options = Options()
    options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    driver.get("https://www.imdb.com/chart/top/")
    time.sleep(5)

    movies = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
    imdb_data = []

    rank = 1
    for movie in movies:
        title_elem = movie.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
        title = title_elem.split(". ")[-1]
        year = movie.find_element(By.CSS_SELECTOR, "span.ipc-metadata-list-summary-item__li").text
        rating = movie.find_element(By.CSS_SELECTOR, "span.ipc-rating-star--rating").text
        imdb_data.append([rank, title, year, rating])
        rank += 1

    driver.quit()
    df_imdb = pd.DataFrame(imdb_data, columns=["Rank", "Title", "Year", "IMDB Rating"])
    df_imdb.to_csv("imdb_top250.csv", index=False)
    print("IMDB Top 250 data saved to imdb_top250.csv")

# ------------------ Q3: Weather Scraper ------------------
def scrape_weather():
    url = "https://www.timeanddate.com/weather/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    table = soup.find("table", class_="zebra fw tb-wt zebra va-m")
    rows = table.find_all("tr")
    weather_data = []

    for row in rows[1:]:  # Skip header row
        cols = row.find_all("td")
        if len(cols) >= 3:
            city = cols[0].text.strip()
            temperature = cols[1].text.strip()
            condition = cols[2].text.strip()
            weather_data.append([city, temperature, condition])

    df_weather = pd.DataFrame(weather_data, columns=["City", "Temperature", "Condition"])
    df_weather.to_csv("weather.csv", index=False)
    print("Weather data saved to weather.csv")

if __name__ == "__main__":
    print("Starting scraping tasks...")
    scrape_books()
    scrape_imdb()
    scrape_weather()
   
