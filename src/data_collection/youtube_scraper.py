import requests
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import (
    TimeoutException, ElementClickInterceptedException,
    ElementNotInteractableException, NoSuchElementException,
)
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

import os
from datetime import datetime
import pandas as pd
import numpy as np

# Multithreading
from concurrent.futures import ThreadPoolExecutor

used_links_path = 'data/raw/used_links.txt'
channel_urls = [
    'https://www.youtube.com/c/TheYoungTurks/videos',
    'https://www.youtube.com/c/TheDailyWire/videos',
    'https://www.youtube.com/c/C-SPAN/videos',
    'https://www.youtube.com/c/VisualPolitikEN/videos',
    'https://www.youtube.com/c/TheDailyShow/videos',
    'https://www.youtube.com/c/SamSeder/videos',
    'https://www.youtube.com/c/StevenCrowder/videos',
    'https://www.youtube.com/c/VisualPolitikTV/videos',
    'https://www.youtube.com/c/NextNewsNetwork/videos',
    'https://www.youtube.com/c/WhiteHouse/videos',
    'https://www.youtube.com/c/markdice/videos',
    'https://www.youtube.com/c/thedavidpakmanshow/videos',
    'https://www.youtube.com/c/AnthonyBrianLogan/videos',
    'https://www.youtube.com/c/JonathanPie/videos',
    'https://www.youtube.com/c/amtv/videos',
    'https://www.youtube.com/c/CaspianReport/videos',
    'https://www.youtube.com/c/ChrisRayGun/videos',
    'https://www.youtube.com/c/FreedomToons/videos',
]


# Scrape channels
def scrape_channel(url):
    print(f'Started scraping {url}')
    options = Options()
    options.headless = True

    chrome_webdriver_path = '/home/ilolio/Documents/chromedriver'
    driver = webdriver.Chrome(chrome_webdriver_path, options=options)

    driver.get(url)
    i = 0
    i_max = 100  # Set the number of times to scroll the page down
    while i < i_max:
        driver.execute_script(f"window.scrollTo(0, {i + 1}*9999);")
        time.sleep(0.4)
        i += 1
        print(f'The page was scrolled {i} of {i_max} times.', end='\r', flush=True)

    time.sleep(15)
    soup = BeautifulSoup(driver.page_source, 'lxml')

    links = list(map(lambda a_tag: f'https://www.youtube.com{a_tag["href"]}',
                     soup.find_all('a', {'class', 'yt-simple-endpoint style-scope ytd-grid-video-renderer'})))
    headlines = list(map(lambda a_tag: a_tag.text,
                         soup.find_all('a', {'class', 'yt-simple-endpoint style-scope ytd-grid-video-renderer'})))
    author = re.findall(r"\.com/c/(.*?)/videos", url)[0]

    # Deleting used links
    new_links = []
    new_headlines = []
    if os.path.exists(used_links_path):
        with open(used_links_path) as f:
            used_links = list(f.read().splitlines())
            for link, headline in zip(links, headlines):
                if link not in used_links:
                    new_links.append(link)
                    new_headlines.append(headline)


    driver.close()
    print('\n\nThe channel is scraped.\n')
    return new_links, new_headlines, author


for url in channel_urls:
    # Scrape videos' stats
    headers = {
        # 'Referer': 'http://www.google.com/',
        # 'Accept': 'test/static,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        # 'Accept-Encoding': 'br, gzip, deflate',
        'Accept-Language': 'en-gb',
        # 'user-agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'
    }
    videos_scraped = 0


    def scrape_video(url):
        global videos_scraped
        r = requests.get(url, headers=headers)

        soup = BeautifulSoup(r.text, 'lxml')

        scripts = soup.find_all('script')

        for script in scripts:
            if 'var ytInitialData' in str(script.string):
                tag = script
                break
        try:
            data = tag.string[20:-1]
        except:
            print('No script tag with the data found')
            return

        try:
            date = re.findall(r'"dateText":{"simpleText":"(.*?)"}', data)[0]
        except:
            date = None
        try:
            views = int(''.join(
                re.findall(r'"viewCount":{"videoViewCountRenderer":{"viewCount":{"simpleText":"(\d+,?\d+) views"}',
                           data)[
                    0].split(',')))
        except:
            views = 0
        try:
            likes = int(
                ''.join(re.findall(r'"accessibilityData":{"label":"(\d+,?\d+) likes"}}', data)[0].split(',')))
        except:
            likes = 0
        try:
            dislikes = int(
                ''.join(re.findall(r'"accessibilityData":{"label":"(\d+,?\d+) dislikes"}}', data)[0].split(',')))
        except:
            dislikes = 0

        videos_scraped += 1
        print(f"{videos_scraped} of {len(links)} videos scraped. {(len(links) - videos_scraped) / 19} seconds left",
              end='\r', flush=True)
        return views, likes, dislikes, date


    # Execution
    # url = 'https://www.youtube.com/c/TheYoungTurks/videos'
    start_channel = time.perf_counter()
    try:
        links, headlines, author = scrape_channel(url=url)
    except Exception as e:
        print(e)
        continue
    end_channel = time.perf_counter()
    print(end_channel - start_channel,
          '- seconds to scrap a channel scrolled 1400 times\n')  # ~0.61 scroll per second

    print('Videos scraping started.')
    start_videos = time.perf_counter()
    try:
        with ThreadPoolExecutor() as executor:
            stats = list(executor.map(scrape_video, links))
            print('\nAll the videos were successfully scraped.\n')
        end_videos = time.perf_counter()
        print(f'\nAll the {len(links)} videos were successfully scraped in {end_videos - start_videos} seconds.\n')  # ~18 videos per second

        # Saving data
        columns = ['headline', 'author', 'views', 'likes', 'dislikes', 'date']
        data = pd.DataFrame(data=[[headline, author, *stat] for headline, stat in zip(headlines, stats)],
                            columns=columns)

        scraping_date = datetime.utcnow().strftime("%Y-%m-%d")  # Use to have one saving_path for each day
        saving_path = f'data/raw/{scraping_date}.csv'
        data.to_csv(saving_path, mode='a', index=False, header=not os.path.exists(saving_path))

        # Adding scraped links to used links list
        mode = 'a' if os.path.exists(used_links_path) else 'w'
        with open(used_links_path, mode=mode) as f:
            f.write('\n'.join(links) + '\n')
    except Exception as e:
        print(e)
        continue
