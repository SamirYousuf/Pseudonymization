# Swell project on de-identification

The goal of this work is to de-identify the personal data such as _name, personal number, phone number etc._ The data is either psuedonymise or anonymise according to the information in the text. Most of the de-identification is done with respect to the grammar so that the resulted data is meaningful. Initially, the data is tokenized using NLTK sentence tokenizer as it is easy to replace accordingly without messing the grammar.

### Requirements and implementation command

- Requirements
  - Python
  - Numpy
  - NLTK
  - Pandas
  - Regular Expressions
  - Levenshtein Distance
 
- Command - 
  _python3 main_file.py --input input.txt --output output.txt_
  
### Sub program

- cities_countries.py
  - Replace the cities and countries in the text
  - _example1 - "I live in Paris, France." will change to "I live in London, England._"
  - _example2 - "He lives in Gothenburg. His family lives in London, England." with change to "He lives in Gothenburg. His family lives in Paris, France."
  
- city_island.py
  - every city in sweden is replaced with an other city
  - every island in sweden is repalced with an other island
  - above two points because city is preceding with "_in_" and island is preceding with "_on_"
  - _example - "In Gothenburg" changes to "In Stockholm" whereas "On Hisingen" changes to "On Vrango"
  
- ids_dates.py
  - Anonymise the _vehicle registeration number_ (only Swedish)
  - _Phone number_ - mobile, landline (only Swedish)
  - _Date formats_ that are mostly used in various parts of the world 
  - 1111/11/11
  - 11/11/11
  - 111111
  - 11.11.11
  - 11/11
  - _Personel Number_ format (only Swedish)
  - 123456-0000
  - 19123456-0000
  - 1234560000
  - 191234560000
  - _Email addresses_ are changed to "email@dot.com"
  - _Website_ and URL are changed to "url.com" except person website
  - Person website is complicated to anonymise because there are thousands of domain to look for
  - "personname.xx"
  - Anonymise the _bank account_
  
- names.py
  - replace the names in the data with random names
  - this program depends on what names dataset is used
  - it will be better if the names are seperate for male and female
  
- siblings_age_family.py
  - age is randomised with (-2,2) for the given age
   - age in digits such as _I am 20._ 
   - age in words such as _I am twenty._
   - age when misspelled such as _I am twent._ which will be corrected using Levenshtein distance and randomised later
  - family members are randomised using a list of relationships
  - friends, brothers and sisters are plural which is not replaced but number of is randomised
  
- swedish_city_street.py
  - this program changes the cities and streets mentioned in the data
  - _example - "I live in Vasastan, Gothenburg" will be changed to "I live in Kungstorget, Stockholm"_
  - the program also replace the swedish postal code such as _"123 45" to "000 00"_
  
- transport.py
  - every private vehicle is randomised using a list of private vehicles
  - stations and tranportation stop are replace such as _"busshållplats" to "tågstation"
  - more list can be added if someone need to randomise more data
  
- university_prof.py
  - universities mention in the data are randomised using a list of universities (_only Swedish_)
  - professional work and studies are replace using swedish jobs and studies database
  
### Main file

main_file.py have the code for all the above mentioned task in sub_program

### Data

csv files below have data in pandas dataframe format

- cities_sweden.csv    _# total cities in sweden_
- city_country.csv     _# every city and country in the world_
- city_street.csv      _# swedish cities and street names (uncompleted)_
- country_capital.csv  _# countries and there capital city dataset_
- island_sweden.csv    _# swedish islands_
- language.csv         _# dataset with every language in the world_
- Prof_dataset.csv     _# swedish professional jobs dataset






### Author
_**Samir**_
