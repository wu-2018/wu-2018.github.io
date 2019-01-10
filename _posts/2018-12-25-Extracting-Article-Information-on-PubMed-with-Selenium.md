---
layout: post  
author: Qinkai WU
title: Extracting Article Information from PubMed with Selenium  
abstract: Using selenium for automatically searching research articles and extracting the key messages from pubmed!  
uniform_img_size: 800px  
id: 2018112501  
tags: ['Selenium', 'Web Crawler']
---

Recently I was assigned some unpleasant chores - reading papers and searching for the key information, then filling a form. The main unwillingness derives from the fact that these papers are typically some cohort studies, which was not relevant to my experience or interest at all.  
  
![form](/img/blog/2018122501/table.png)
  
So I was thinking if there's any way that I can make it more automated.  
Quickly I got `Selenium` in my mind. Compared to other python crawler packages like `requests` and `scrapy`, Selenium actually drives a 'human' browser (like IE, Chrome, Firefox, and so on), so we can see how it works, how the webpages are scrolled and clicked, i.e., we can supervise the entire process more intuitively.  

> #### Note that the following instruction is for Firefox, which is the default in Ubuntu.  
> #### The corresponding Jupyter Notebook is available [here at my github](https://github.com/wu-2018/Collections/tree/master/selenium_pubmed).  


## Installation

Installing selenium and checking its version:

```shell
sudo pip3 install selenium

python3 -c "import selenium;print(selenium.__version__)"
```
[Click here to check the version requirements between Selenium, Firefox and geckodriver.](http://firefox-source-docs.mozilla.org/testing/geckodriver/geckodriver/Support.html)  
  
![version](/img/blog/2018122501/version_req.png)
  
Then download [geckodriver](https://github.com/mozilla/geckodriver/releases).  
  
Finally,  
```shell
tar -zxvf geckodriver*.tar.gz
chmod +x geckodriver
sudo mv geckodriver /usr/bin/geckodriver
```

## Starting the task

First, save all the paper titles in a plain text file **(one title per line)** like this:  
  
![title_example](/img/blog/2018122501/title_e.png)

Reading file and saving all titles in a list:


```python
with open("article_title.txt", "r") as f:
    paper_list = f.readlines()

paper_list = [i.strip().replace("\xa0", " ") for i in paper_list if i.strip()]
paperList = list(set(paper_list))
print(len(paperList))
paperList = sorted(paperList, key=lambda x:paper_list.index(x))
```

    11



```python
import os
from collections import defaultdict

from selenium import webdriver
```

Specify the path for downloaded pdf:


```python
download_dir = "pdf_downloaded"

if not os.path.exists(download_dir):
    os.mkdir(download_dir)
```

Some settings like disabling image, css and javascript loading, directly saving the pdf file rather than opening it...


```python
profile = webdriver.FirefoxProfile()
profile.set_preference('permissions.default.stylesheet', 2)
profile.set_preference('permissions.default.javascript', 2)
profile.set_preference('permissions.default.image', 2)
profile.set_preference('browser.download.folderList', 2)
profile.set_preference('browser.download.dir', os.path.abspath(download_dir))
profile.set_preference('browser.download.manager.showWhenStarting', False)
profile.set_preference('browser.helperApps.neverAsk.saveToDisk', 'application/pdf')
profile.set_preference('pdfjs.disabled', True)

browser = webdriver.Firefox(firefox_profile=profile)
```

Now a browser window would pop up!  
  
Notice the little robot symbol in the address bar, that means the browser is under the control of exterior program.  
  
![firfox_robot](/img/blog/2018122501/firefox_robot.png)

In most cases, upon sending the request we can directly get to the page with detailed information, so just `F12` and look for the corresponding tags then extract them using the method like `.find_element_by_class_name()`.  
  
![de](/img/blog/2018122501/detail.png)

But there are still some other cases where things can get much more troublesome:  
  
Multiple results:  
  
![mul](/img/blog/2018122501/multiple_result.png)
  
Fortunately, sometimes there's a suggestion box and it's highly possible that's just the correct match:  
  
![sensor](/img/blog/2018122501/sensor_content.png)  
  
So just let selenium click that link in `sensor_content` `div`!  

Anyway, we have to write extra rules to handle these exceptions. And the `similar_results_judge()` below is just an example.   


```python
total_di = defaultdict(defaultdict)
pubmed_url_base = "https://www.ncbi.nlm.nih.gov/pubmed/"

def similar_results_judge():
    try:
        item = browser.find_element_by_class_name("result_count")
        if int(item.text.split()[-1]) >1:
            print(item.text)
            try:
                correct_paper = browser.find_element_by_class_name("sensor_content")
                correct_paper.find_elements_by_tag_name("a")[-1].click()
                print("`sensor_content` found! Jumping to the detailed page..." )
            except:
                print("`sensor_content` not found. Trying again using simple mode..." )
                return True
    except:
        pass
```

Main function for loading webpages and extracting needed information:  


```python
def search_pubmed(i, a, di, title_mode=True):
    # search papers in pubmed
    if title_mode:
        pubmed_ = pubmed_url_base + "?term=" + a + "[Title]"
    else:
        pubmed_ = pubmed_url_base + "?term=" + a
    
    #browser.set_page_load_timeout(100)
    try:
        browser.get(pubmed_)
        
        if similar_results_judge():
            # trying searching without specifying the '[Title]' constrain
            search_pubmed(i, a, di, title_mode=False)

        di[i]['title'] = a

        cit = browser.find_element_by_class_name("cit")
        di[i]['desc'] = cit.text
        di[i]['journal'] = cit.find_element_by_tag_name('a').text

        aux =  browser.find_element_by_class_name("aux")
        dt = aux.find_elements_by_tag_name("dt")
        dd = aux.find_elements_by_tag_name("dd")
        
        for ii,v in enumerate(dt):
            di[i][v.text] = dd[ii].text
            
        abstr = browser.find_element_by_class_name("abstr")        
        di[i]['abstr'] = abstr.text

        #print(di[i])
    except Exception as e:
        print("Error!",e)
    finally:
        print("processed %s"%i)
        print("-"*20)
```

Sometimes `Timeout` problem can be quite often. So set a `max_iter` to request the omitted papers multiple times.  


```python
for i,a in enumerate(paperList):
    search_pubmed(i, a, total_di)
    
max_iter = 5
iter = 0
while iter < max_iter:
    rest = set(range(len(paperList))) - set(total_di.keys())
    if not rest:
        break
    print(rest)
    for i in rest:
        search_pubmed(i, paperList[i], total_di)
    iter += 1
```

    processed 0
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 1
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 2
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 3
    --------------------
    processed 4
    --------------------
    processed 5
    --------------------
    processed 6
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 7
    --------------------
    Items: 1 to 20 of 287
    `sensor_content` not found. Trying again using simple mode...
    Items: 1 to 20 of 287
    `sensor_content` found! Jumping to the detailed page...
    processed 8
    --------------------
    processed 8
    --------------------
    processed 9
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 10
    --------------------
    {10, 3, 7}
    Error! Message: Timeout loading page after 300000ms
    
    processed 10
    --------------------
    processed 3
    --------------------
    Error! Message: Timeout loading page after 300000ms
    
    processed 7
    --------------------
    {10, 7}
    processed 10
    --------------------
    processed 7
    --------------------


Download the pdf file if it's publicly available through PMC.  


```python
ncbi_url = "https://www.ncbi.nlm.nih.gov/"

def download_pmc_pdf(di, pmcid_key):
    
    for i,d in di.items():
        if pmcid_key in d.keys():
            
            pmc_url = ncbi_url + "pmc/articles/" + d[pmcid_key] + "/"

            browser.get(pmc_url)
            try:
                down_ = browser.find_element_by_class_name("format-menu")
                pdf_link = down_.find_elements_by_tag_name("a")[2]
                # starting to download pdf
                pdf_link.click()

                pdf_url = pdf_link.get_attribute('href')
                filename = pdf_url.split('/')[-1]
                status = os.path.exists(filename)

                di[i]['pdf_url'] = pdf_url
                di[i]['filename'] = filename if status else ""

            except Exception as e:
                print(e)
                
download_pmc_pdf(total_di, 'PMCID:')
```

Check available information for each article:  
```python
for k,d in total_di.items():
    print(k, d.keys())
```

    0 dict_keys(['title', 'desc', 'journal', 'PMID:', 'abstr'])
    1 dict_keys(['title', 'desc', 'journal', 'PMID:', 'abstr'])
    2 dict_keys(['title', 'desc', 'journal', 'PMID:', 'DOI:', 'abstr'])
    4 dict_keys(['title', 'desc', 'journal', 'PMID:', 'DOI:', 'abstr'])
    5 dict_keys(['title', 'desc', 'journal', 'PMID:', 'PMCID:', 'DOI:', 'abstr', 'pdf_url', 'filename'])
    6 dict_keys(['title', 'desc', 'journal', 'PMID:', 'PMCID:', 'DOI:', 'abstr'])
    8 dict_keys(['title', 'desc', 'journal', 'PMID:', 'DOI:', 'abstr'])
    9 dict_keys(['title', 'desc', 'journal', 'PMID:', 'PMCID:', 'DOI:', 'abstr', 'pdf_url', 'filename'])
    3 dict_keys(['title', 'desc', 'journal', 'PMID:', 'DOI:', 'abstr'])
    10 dict_keys(['title', 'desc', 'journal', 'PMID:', 'DOI:', 'abstr'])
    7 dict_keys(['title', 'desc', 'journal', 'PMID:', 'abstr'])

Close the browser.  
```python
browser.quit()
```

Save dict into pickle file.
```python
import pickle
with open("papers_info.pickle", "wb") as f:
    pickle.dump(total_di, file=f)
```

So, now we have the `doi`, `abstract` and many other useful information. 

But what about their `experientment design`, `results` and ...? Sorry. Maybe NLP would help? However certainly that's far beyond my ability currently!  


