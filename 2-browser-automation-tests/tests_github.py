# -*- coding: utf-8 -*-
"""
Some exercise tests on GitHub website. The tests are organised through the 
unittest testing module and structured with the 'page object model'

@author: Andrea Boselli
"""

#%% Import required Python modules

import time
import unittest

from collections                     import Counter
from datetime                        import datetime
from datetime                        import date
from selenium                        import webdriver
from selenium.common.exceptions      import NoSuchElementException
from selenium.webdriver.common.by    import By
from selenium.webdriver.common.keys  import Keys
from selenium.webdriver.support.wait import WebDriverWait


#%% Execution settings

driver_path = '' # TODO: set the webdriver path

tests_to_try = ['test_4']   # executed tests

wait_time    = 2            # waiting time in seconds after some operations in the tests
search_str   = 'Test'       # searched expression in the tests
min_n_contrs =  50          # minimum number of contributions for TEST2
min_stars    = 100          # minimum number of total stars in the first results page for TEST4 
dummy_date   = '9000-01-01' # reference date, in case of errors in dates parsing 


#%% Page Object Model classes

class BasePage:
    """Page with the common features and operations of the pages of GitHub website"""
    
    def __init__(self, driver):
        self.driver = driver


class HomePage(BasePage):
    """Homepage of GitHub website"""
    
    def __init__(self,driver):
        super().__init__(driver)

    def search(self,search_input):
        """Search on the website"""
        self.driver.find_element(By.CSS_SELECTOR, "input[type='text']").send_keys(search_input + Keys.ENTER)
        

class ResultsPage(BasePage):
    """Results page of GitHub"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def click_group_item(self,group_name):
        """Click on the grouping item with the specified name and wait for it to be fully loaded"""
        self.driver.find_element(By.CSS_SELECTOR, ".menu-item > [data-search-type='" + group_name + "']").click()
        WebDriverWait(self.driver, 10).until(lambda driver : driver.find_element(By.CSS_SELECTOR, ".menu-item.selected > [data-search-type='" + group_name + "']"))
        
    def go_to_last_page(self):
        """Go to the last result page for the current search and wait for it to be fully loaded"""
        self.driver.find_element(By.CSS_SELECTOR, "[role='navigation'] > a:nth-last-child(2)").click()
        WebDriverWait(self.driver, 10).until(lambda driver : driver.find_element(By.CSS_SELECTOR, "[role='navigation'] > .next_page.disabled"))
        
    def click_last_repo(self):
        """Click on the last repository on the current page"""
        self.driver.find_element(By.CSS_SELECTOR, ".repo-list > .repo-list-item:last-child .text-normal > a").click()
        
    def get_users_links(self):
        """Get the links to the users pages on the current page"""
        users_elems = self.driver.find_elements(By.CSS_SELECTOR, "#user_search_results .Box .text-normal > a:nth-child(1)")
        return [elem.get_attribute('href') for elem in users_elems]
    
    def check_next_page(self):
        """Check the availability of a next page with respect to the current one"""
        return len(self.driver.find_elements(By.CSS_SELECTOR, "[role='navigation'] > .next_page.disabled")) != 1
        
    def go_to_next_page(self, wait_time=2):
        """Go to the next page with respect to the current one and wait for wait_time"""
        self.driver.find_element(By.CSS_SELECTOR, "[role='navigation'] > .next_page").click()
        time.sleep(wait_time)
        
    def get_commits_links(self):
        """Get the links to the commits on the current page"""
        commits_elems = self.driver.find_elements(By.CSS_SELECTOR, "#commit_search_results > div div.text-normal > a")
        return [elem.get_attribute('href') for elem in commits_elems]
    
    def get_repos_links(self):
        """Get the links to the repositories pages on the current page"""
        repos_elems = self.driver.find_elements(By.CSS_SELECTOR, ".repo-list > .repo-list-item a.v-align-middle")
        return [elem.get_attribute('href') for elem in repos_elems]
        
        
class RepoPage(BasePage):
    """Repository page on GitHub"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def get_last_commit_date(self, dummy_date):
        """Get the string with the date of the last commit; return a dummy date in case of no commits"""
        try:
            last_commit_date = self.driver.find_element(By.CSS_SELECTOR, ".Box-header relative-time").get_attribute("datetime")
        except NoSuchElementException:
            return dummy_date
        else:
            return last_commit_date
        
    def get_n_stars(self):
        """Get the number of stars of the current repository page"""
        return int(self.driver.find_element(By.ID, "repo-stars-counter-star").get_attribute('title').replace(',',''))
    
    def get_languages(self):
        """Get programming languages used in the current repository page"""
        languages_elems = self.driver.find_elements(By.XPATH, "//h2[text()='Languages']/../ul/li")
        return [elem.text.split('\n')[0] for elem in languages_elems]


class UserPage(BasePage):
    """User page on GitHub"""
    
    def __init__(self,driver):
        super().__init__(driver)
        
    def get_n_contributions(self):
        """Get the number of contributions of a user (return 0 in case of an organization)"""
        try:
            n_contr_text = self.driver.find_element(By.XPATH, "//h2[contains(text(),'contribution')]").text
        except NoSuchElementException:
            return 0
        else:
            return int(n_contr_text.split(' ')[0].replace(',',''))
        
    def get_nickname(self):
        """Get the nickname of a user (return 0 in case of an organization)"""
        return  self.driver.find_element(By.CSS_SELECTOR, "span.p-nickname").text
        

#%% Tests with unittest module

class GitHub_tests(unittest.TestCase):
    """Extends the unittest.TestCase class and implements through page object
       model the required automated tests on GitHub website"""
    
    def setUp(self):
        self.driver = webdriver.Chrome(executable_path=driver_path)
        self.driver.implicitly_wait(3)
        self.driver.get("https://github.com/")
    
    
    def tearDown(self):
        self.driver.quit()
    
    
    def test_1(self):
        """Search in the repositories, go to the last page, open the last repo, 
           check that the last commit date (if any) is before the current date. 
           The test should fail in case of no commits"""
        
        # Extract the date of the last repository
        HomePage(self.driver).search(search_str)
        resultsPage = ResultsPage(self.driver)
        resultsPage.go_to_last_page()
        resultsPage.click_last_repo()
        last_commit_date = RepoPage(self.driver).get_last_commit_date(dummy_date)
        print("TEST1 - Last commit date: ", last_commit_date)
        
        # Check the required condition on the extracted date
        last_commit_date = datetime.strptime(last_commit_date.split('T')[0], "%Y-%m-%d").date()
        self.assertTrue(last_commit_date < date.today())
        
    
    def test_2(self):
        """Among the users, search the first one with at least 50 contributions
           in the last year: check whether the name of this user has at least 10
           characters and it contains the letter 'a'"""
        HomePage(self.driver).search(search_str)
        resultsPage = ResultsPage(self.driver)
        resultsPage.click_group_item('Users')
        results_window = self.driver.current_window_handle
        
        # Iterate on each result page
        search_curr_page = True
        user_found       = False
        while(search_curr_page):
            
            # Iterate on each user page
            users_links = resultsPage.get_users_links()
            for link in users_links:
                
                # Open current user link on a new window
                self.driver.execute_script("window.open('{0}')".format(link))
                for window_handle in self.driver.window_handles:
                    if window_handle != results_window:
                        self.driver.switch_to.window(window_handle)
                        break
                time.sleep(wait_time)
                
                # Perform the required operations on the current user
                userPage = UserPage(self.driver)
                n_contributions = userPage.get_n_contributions()
                print("TEST2 - Contributions of the current user: ", n_contributions)
                if(n_contributions >= min_n_contrs):
                    user_found    = True
                    user_username = userPage.get_nickname()
                    print("TEST2 - Username of the found user: ", user_username)
                
                # Go back to the original window
                self.driver.close()
                self.driver.switch_to.window(results_window)
                if(user_found):
                    break
                
            # Check whether to perform the next page iteration
            if(resultsPage.check_next_page() and not user_found):
                resultsPage.go_to_next_page(wait_time)
            else:
                search_curr_page = False
                
        self.assertTrue(user_found and (len(user_username) > 10) and ('a' in user_username))
        
    
    def test_3(self):
        """List the programming languages employed in the repositories of the first 
           results page and the number of repos in which they are used"""
        
        # Get the list of repos links of the first page
        HomePage(self.driver).search(search_str)
        resultsPage = ResultsPage(self.driver)
        resultsPage.click_group_item('Repositories')
        repos_links = resultsPage.get_repos_links()
        results_window = self.driver.current_window_handle
        
        # Iterate on each repository page
        repos_languages = []
        for link in repos_links:
            
            # Open current repo link on a new window
            self.driver.execute_script("window.open('{0}')".format(link))
            for window_handle in self.driver.window_handles:
                if window_handle != results_window:
                    self.driver.switch_to.window(window_handle)
                    break
            time.sleep(wait_time)
            
            # Get the programming languages of the current repository
            repos_languages.extend(RepoPage(self.driver).get_languages())
            self.driver.close()
            self.driver.switch_to.window(results_window)
        
        # Test outcome
        repos_languages = Counter(repos_languages)
        print("TEST3 - The retrieved languages are:\n", repos_languages)
        
    
    def test_4(self):
        """Search among the repositories, and check that the sum of the stars of 
           all the repositories in the first page is greater than 100"""
        
        # Get the list of repos links of the first page
        HomePage(self.driver).search(search_str)
        resultsPage = ResultsPage(self.driver)
        resultsPage.click_group_item('Repositories')
        repos_links = resultsPage.get_repos_links()
        results_window = self.driver.current_window_handle
        
        # Iterate on each repository page
        repos_stars = []
        for link in repos_links:
            
            # Open current repo link on a new window
            self.driver.execute_script("window.open('{0}')".format(link))
            for window_handle in self.driver.window_handles:
                if window_handle != results_window:
                    self.driver.switch_to.window(window_handle)
                    break
            time.sleep(wait_time)
            
            # Get the stars number of the current repository
            repos_stars.append(RepoPage(self.driver).get_n_stars())
            self.driver.close()
            self.driver.switch_to.window(results_window)
        
        # Test outcome
        total_stars = sum(repos_stars)
        print("TEST4 - The stars at the first page are:\n", repos_stars)        
        print('TEST4 - The total number of stars in the first page is: ', total_stars)
        self.assertTrue(total_stars > min_stars)
        
        
#%% Tests run

if __name__ == '__main__':
    
    suite = unittest.TestSuite()
    
    for test_to_try in tests_to_try:
        suite.addTest(GitHub_tests(test_to_try))
    
    runner = unittest.TextTestRunner()
    runner.run(suite)