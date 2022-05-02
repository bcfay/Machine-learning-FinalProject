import selenium
from selenium import webdriver
#from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import pandas as pd

def scrape(letter):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    link = "https://worldwide.espacenet.com/classification?locale=en_EP#!/CPC=" + letter
    print(link)
    driver.get(link)
    #classes = driver.find_elements(by=By.XPATH, value="//a[@class=\"symbol classref\"]")

    '''
    classes = driver.find_elements(by=By.CSS_SELECTOR, value='a.symbol.classref')
    titles = driver.find_elements(by=By.CSS_SELECTOR, value='span.raw-text')
    #a = driver.find_element_by_css_selector()
    #print(classes)
    print(classes)
    for element in classes:
        print(element.text)

    print(titles)
    for element in titles:
        print(element.text)
    '''
    sections = driver.find_elements(by=By.CSS_SELECTOR, value='div.classitem.clearfix.level-4.has-children')
    for each_section in sections:
        #children = each_section.find_elements(by=By.CSS_SELECTOR, value='./child::*')
        classes = each_section.find_elements(by=By.CSS_SELECTOR, value='a.symbol.classref')
        titles = each_section.find_elements(by=By.CSS_SELECTOR, value='span.raw-text')
        for a_class in classes:
            print(a_class.text)
        for title in titles:
            print(title.text)

        # children should be the context code, its text, etc
        #for child in children:
        #    print(child.getTagName())

if __name__ == "__main__":
    scrape('B')





