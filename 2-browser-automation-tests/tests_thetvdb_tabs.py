# -*- coding: utf-8 -*-
"""
Some test automation exercises on the webpage 'https://thetvdb.com/'. The tests 
are organised through the unittest testing module and structured with the 'page
object model'. Differently from 'Tests_TheTVDB.py' script, when multiple pages
of the same kind need to be opened, they are opened in new tabs.

@author: Andrea Boselli
"""

#%% Import required Python modules

import unittest
import numpy as np

from datetime                        import datetime
from selenium                        import webdriver
from selenium.webdriver.common.by    import By
from selenium.webdriver.common.keys  import Keys


#%% Execution settings

driver_path = '' # TODO: set the webdriver path

tests_to_try = ['test_4','test_6']            # executed tests

series_to_find = "The Bold and the Beautiful" # considered series
min_episodes   = 10                           # minimum required number of episodes per season

series_list = ["Breaking Bad",
               "The Bold and the Beautiful"]  # series among which to detect the one with more episodes

dummy_date = datetime.strptime("January 1, 9000", "%B %d, %Y") # reference date, in case of errors in dates parsing


#%% Page Object Model classes

class BasePage:
    """Page with the common features and operations of the pages of TheTVDB website"""
    
    def __init__(self, driver):
        self.driver = driver


class HomePage(BasePage):
    """Homepage of TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)

    def search(self,title):
        """Search for the input series"""
        self.driver.find_element(By.NAME, 'query').send_keys(title + Keys.ENTER)
        

class ResultsPage(BasePage):
    """Series results page of TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def click_first_result(self):
        """Click on the first series in the results page"""
        self.driver.find_element(By.TAG_NAME, "h3").find_element(By.TAG_NAME, "a").click()


class SeriesPage(BasePage):
    """Specific series page of TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def click_section_button(self,section):
        """Click on a specific section button"""
        self.driver.find_element(By.LINK_TEXT, section).click()
        
    def get_n_posters(self):
        """Get the number of posters for the series"""
        return self.driver.find_element(By.PARTIAL_LINK_TEXT, "Poster").find_element(By.TAG_NAME, "span").text
    
    def get_n_actors(self):
        """Get the number of actors for the series"""
        return self.driver.find_element(By.CSS_SELECTOR, "#castcrew a[href='#people-actor'] > span").text
    
    def get_n_seasons(self):
        """Get the number of main seasons for the series (the extra seasons are excluded)"""
        return len( self.driver.find_elements(By.XPATH,"//div[@id='seasons-official']/table/tbody/tr/td[contains(.,'Season ')]/a") )
    
    def get_nth_character_elem(self,n):
        """Get the webelement of the n-th character (starting from 1)"""
        return self.driver.find_element(By.CSS_SELECTOR, "#people-actor > div > div:nth-of-type({0}) a".format(n))
    
    def get_nth_character_link(self,n):
        """Get the link to the n-th character page (starting from 1)"""
        return self.get_nth_character_elem(n).get_attribute('href')
    
    def get_nth_season_elem(self, n):
        """Get the webelement of the n-th main season for the series (the extra seasons are excluded; seasons count starts from 0)"""
        return self.driver.find_elements(By.XPATH,"//div[@id='seasons-official']/table/tbody/tr/td[contains(.,'Season ')]/a")[n]
    
    def get_nth_season_name(self, n):
        """Get the name of the n-th main season for the series (the extra seasons are excluded; seasons count starts from 0)"""
        return self.get_nth_season_elem(n).text.strip()
    
    def get_nth_season_link(self,n):
        """Get the link to the n-th main season for the series (the extra seasons are excluded; seasons count starts from 0)"""
        return self.get_nth_season_elem(n).get_attribute('href')
    
    def get_seasons_n_episodes(self):
        """For each season, extract the number of episodes"""
        first_table  = self.driver.find_element (By.CSS_SELECTOR, "table[class='table table-bordered table-hover table-colored']")
        episodes_col = first_table.find_elements(By.CSS_SELECTOR, "tbody > tr > td:nth-of-type(4)")
        return [season.text for season in episodes_col]
    

class CharacterPage(BasePage):
    """Specific character page of a series on TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def click_actor(self):
        """Go to the actor page"""
        self.driver.find_element(By.XPATH,"//strong[text()='Played By']/../span").click()

    
class ActorPage(BasePage):
    """Specific actor page of TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def get_gender(self):
        """Retrieve the gender of the actor"""
        try:
            gender = self.driver.find_element(By.XPATH,"//strong[text()='Gender']/../span").text
        except:
            gender = 'Unspecified'
        return gender
    
    def get_name(self):
        """Retrieve the name of the actor"""
        return self.driver.find_element(By.ID,'series_title').text


class SeasonPage(BasePage):
    """Specific season page of a series on TheTVDB website"""
    
    def __init__(self,driver):
        super().__init__(driver)
    
    def retrieve_episodes_dates(self, dummy_date):
        """Store the episodes dates of this season in a list; if not available, set a conventional date"""
        season_episodes = []
        table_rows = self.driver.find_elements(By.CSS_SELECTOR, "tbody > tr > td:nth-of-type(3)") # select the rows of the episodes table
        for row in table_rows:
            try:
                row_date = datetime.strptime(row.text.split('\n')[0], "%B %d, %Y") # currently extracted date
            except:
                row_date = dummy_date                                              # assign a dummy date, in case of errors 
            season_episodes.append(row_date) # store the column of the episodes dates
        return season_episodes
        
                                       

#%% Tests with unittest module

class TheTVDB_tests(unittest.TestCase):
    """Extends the unittest.TestCase class and implements through page object
       model the required automated tests"""
    
    def setUp(self):
        self.driver = webdriver.Chrome(executable_path=driver_path)
        self.driver.implicitly_wait(3)
        self.driver.get("https://thetvdb.com/")
    
    
    def tearDown(self):
        self.driver.quit()
    
    
    def test_1(self):
        """Go to the homepage, enter a series name and press the research button"""
        HomePage(self.driver).search(series_to_find)
    
    
    def test_2(self):
        """In the results page, open the searched series and check that it stores at least a poster (Artwork > Poster)"""
        HomePage(self.driver).search(series_to_find)
        ResultsPage(self.driver).click_first_result()
        
        seriesPage = SeriesPage(self.driver)
        seriesPage.click_section_button("Artwork")
        test2_n_posters = seriesPage.get_n_posters()
        
        print("TEST2 - Number of posters in the current series: " + test2_n_posters)
    
    
    def test_3(self):
        """Within a series page, check that each season contains at least min_episodes episodes"""
        HomePage(self.driver).search(series_to_find)
        ResultsPage(self.driver).click_first_result()
        
        seriesPage = SeriesPage(self.driver)
        seriesPage.click_section_button("Seasons")
        test3_episodes_col = seriesPage.get_seasons_n_episodes()
        
        test3_episodes_col_casted = [int(x) for x in test3_episodes_col if x.isdigit()]    # discard the bad strings and convert the remaining ones to integer
        test3_result = all([(x==0)or(x>=min_episodes) for x in test3_episodes_col_casted]) # check the test assertion
        
        print("TEST3 - Episodes column: ")
        print(test3_episodes_col_casted)
        print("TEST3 - Does every series have at least " + str(min_episodes) + " episodes? " + str(test3_result))
        self.assertTrue(test3_result)
        
    
    def test_4(self):
        """In the series page, go to 'Cast & Crew' and find the most frequent gender"""
        HomePage(self.driver).search(series_to_find)
        ResultsPage(self.driver).click_first_result()
        
        seriesPage = SeriesPage(self.driver)
        seriesPage.click_section_button("Cast & Crew")
        test4_n_actors = int(seriesPage.get_n_actors())
        
        test4_genders = []
        series_window = self.driver.current_window_handle
        for n in range(1,test4_n_actors+1): # iterate on each actor
            character_link = seriesPage.get_nth_character_link(n)
            self.driver.execute_script("window.open('{0}')".format(character_link))
            for window_handle in self.driver.window_handles:
                if window_handle != series_window:
                    self.driver.switch_to.window(window_handle)
                    break
            CharacterPage(self.driver).click_actor()
            actorPage = ActorPage(self.driver)
            actor_gender = actorPage.get_gender()
            actor_name   = actorPage.get_name  ()
            test4_genders.append(actor_gender)
            print("TEST4 - Current actor: " + actor_name + ", " + actor_gender)
            self.driver.close()
            self.driver.switch_to.window(series_window)
        
        test4_genders_count = {}
        for gender in set(test4_genders):
            test4_genders_count[gender] = test4_genders.count(gender)
        print("TEST4 - Final genders count:")
        print(test4_genders_count)
        
        
    def test_5(self):
        """Within a list of series, find the one with the highest number of episodes"""
        test5_n_episodes_list = []
        for series in series_list: # Iterate on each series
            HomePage(self.driver).search(series)
            ResultsPage(self.driver).click_first_result()
            
            seriesPage = SeriesPage(self.driver)
            seriesPage.click_section_button("Seasons")
            episodes_col        = seriesPage.get_seasons_n_episodes()
            episodes_col_casted = [int(x) for x in episodes_col if x.isdigit()]
            test5_n_episodes_list.append(sum(episodes_col_casted))
            
            print("TEST5 - Current series: " + series)
            print(episodes_col_casted)
            self.driver.get("https://thetvdb.com/")
        
        max_n_episodes = max(test5_n_episodes_list)
        for n in range(len(series_list)):
            curr_n_episodes = test5_n_episodes_list[n]
            print("TEST5 - '" + series_list[n] + "' has " + str(curr_n_episodes) + " episodes in total")
            if(curr_n_episodes == max_n_episodes):
                print(" "*9 + "and it has the maximal number of episodes")
        
    
    def test_6(self):
        """Given a series, check that all the episodes of each season are chronologically sorted"""
        HomePage(self.driver).search(series_to_find)
        ResultsPage(self.driver).click_first_result()
        
        seriesPage = SeriesPage(self.driver)
        seriesPage.click_section_button("Seasons")
        test6_n_seasons = seriesPage.get_n_seasons()
        print("TEST6 - Number of seasons: " + str(test6_n_seasons))
        
        seasons_names    = [] # names of the seasons
        seasons_episodes = [] # dates of the episodes
        series_window = self.driver.current_window_handle
        for n in range(test6_n_seasons):                    # iterate on the main seasons
            season_name = seriesPage.get_nth_season_name(n) # title of the current season
            season_link = seriesPage.get_nth_season_link(n) # link  to the current season
            self.driver.execute_script("window.open('{0}')".format(season_link))
            for window_handle in self.driver.window_handles:
                if window_handle != series_window:
                    self.driver.switch_to.window(window_handle)
                    break
            season_episodes = np.array(SeasonPage(self.driver).retrieve_episodes_dates(dummy_date)) # NumPy array of the dates of the episodes of the current season
            seasons_names.append(season_name)        # save the season name
            seasons_episodes.append(season_episodes) # save the episodes dates array
            print('TEST6 - ' + season_name + ' has ' + str(len(season_episodes)) + ' episodes')
            self.driver.close()
            self.driver.switch_to.window(series_window)
        
        test6_result = True                   # are all the seasons episodes sorted in the correct order?
        for n in range(test6_n_seasons):
            if(len(seasons_episodes[n]) < 2):        # in case there are less than 2 episodes, it's OK
                continue
            elif(dummy_date in seasons_episodes[n]): # check that all dates are available
                test6_result = False
                print("TEST6 - " + seasons_names[n] + " has some unavailable episode dates")
            elif(any(seasons_episodes[n][1:] < seasons_episodes[n][:-1])): # check that the dates are sorted
                test6_result = False
                print("TEST6 - " + seasons_names[n] + " has some not correctly sorted episodes")
        print("TEST6 - Are the episodes correctly sorted? " + str(test6_result))
        self.assertTrue(test6_result)
        

#%% Tests run

if __name__ == '__main__':
    
    suite = unittest.TestSuite()
    
    for test_to_try in tests_to_try:
        suite.addTest(TheTVDB_tests(test_to_try))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)