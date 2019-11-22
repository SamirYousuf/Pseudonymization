# Modules
# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
from Levenshtein import distance
import argparse, json
from os.path import abspath, dirname

MODULEDIR = dirname(dirname(abspath(__file__)))

# If it is misspelled then the function corrects the spelling and then randomly change to a digit
def get_correct_spelling(word, list_number):
    return min(list_number, key=lambda x: distance(word, x))


# Read list and dict from dict_data 
# File contains university names, relationship in family, days, months etc.
with open(MODULEDIR + '/dataset/dict_data.json', 'r') as file:
    dict_data = json.load(file)
    
dict_numbers = dict_data['dict_numbers']
list_family = dict_data['list_family']
list_siblings = dict_data['list_siblings']
list_days = dict_data['list_days']
list_months = dict_data['list_months']
dict_universities = dict_data['dict_universities']
list_transports = dict_data['list_transports']
list_stations = dict_data['list_stations']
list_stations_en = dict_data['list_stations_en']

with open(MODULEDIR + '/dataset/names_database_1.json', 'r') as file:
        dict_names = json.load(file)

        
list_job_title = pd.read_csv(MODULEDIR + '/dataset/Prof_dataset.csv')
list_data = pd.read_csv(MODULEDIR + '/dataset/city_country_population.csv')
list_data_ = pd.read_csv(MODULEDIR + '/dataset/_city_country_population.csv')
list_swedish_cities = pd.read_csv(MODULEDIR + '/dataset/cities_sweden.csv')
list_swedish_island = pd.read_csv(MODULEDIR + '/dataset/island_sweden.csv')
swe_street_data = pd.read_csv(MODULEDIR + '/dataset/swedish_streets.csv')


# Main function to de-identify all the personal information and save as a text file
def identify(data):
    # To have the output format same as the input especially for newline and paragraphs in the text
    #data = re.sub(r'\n\n', ' $$$$ . ', data)  
    data = nltk.sent_tokenize(data) # Sentence Tokenize to keep track on the 
    
    '''
    ##########
    # In this function a list of personal data is anonymised
    # Anonymise the vehicle registeration number (only Swedish)
    # Phone number - mobile, landline (only Swedish)
    # Date formats that are mostly used in various parts of the world
    # - 1111/11/11
    # - 11/11/11
    # - 111111
    # - 11.11.11
    # - 11/11
    # Personel Number format (only Swedish)
    # - 123456-0000
    # - 19123456-0000
    # - 1234560000
    # - 191234560000
    # Bank format in Sweden
    # - 1234-00 200 00
    # - 1234-123 123 123
    # - 1234-1 123 123 1234
    # Email addresses are changed to 'email@dot.com'
    # Website and URL are changed to "url.com" except person website
    # Person website is complicated to anonymise because there are thousands of domain to look for
    # - "personname.xx"
    ##########
    '''

    indexing = 1
    _data = []
    
    # Swedish bank account format
    for line in data:
        if re.search(r'\b\d{4}-\d{2} \d{3} \d{2}\b', line):
            line = re.sub(r'(\b\d{4}-\d{2} \d{3} \d{2}\b)', '0000-00$€000$€00/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{4}-\d{3} \d{3} \d{3}\b', line):
            line = re.sub(r'(\b\d{4}-\d{3} \d{3} \d{3}\b)', '0000-000$€000$€000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{4}-\d{1} \d{3} \d{3} \d{4}\b', line):
            line = re.sub(r'(\b\d{4}-\d{1} \d{3} \d{3} \d{4}\b)', '0000-0$€000$€000$€0000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{4} \d{2} \d{3} \d{2}\b', line):
            line = re.sub(r'(\b\d{4} \d{2} \d{3} \d{2}\b)', '0000$€00$€000$€00/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{4} \d{3} \d{3} \d{3}\b', line):
            line = re.sub(r'(\b\d{4} \d{3} \d{3} \d{3}\b)', '0000$€000$€000$€000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{4} \d{1} \d{3} \d{3} \d{4}\b', line):
            line = re.sub(r'(\b\d{4} \d{1} \d{3} \d{3} \d{4}\b)', '0000$€0$€000$€000$€0000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{15}\b', line):
            line = re.sub(r'(\b\d{15}\b)', '000000000000000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{13}\b', line):
            line = re.sub(r'(\b\d{13}\b)', '0000000000000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        elif re.search(r'\b\d{11}\b', line):
            line = re.sub(r'(\b\d{11}\b)', '10000000000/label/account_nr/label/'+str(indexing), line)
            _data.append(line)
            indexing += 1
        else:
            _data.append(line)
    
    data = _data
    _data = []
    
    for line in data:
        if re.search(r'\b[A-Z]{3}\b \d{3}\b', line) or re.search(r'\b[A-Z]{3}\b \d{2}[A-Z]{1}\b', line):  # Vehicle License number
            line = re.sub(r'(\b[A-Z]{3}\b \d{3}\b)', 'ABC$€000/label/license_nr/label/'+str(indexing), line)
            line = re.sub(r'(\b[A-Z]{3}\b \d{2}[A-Z]{1}\b)', 'ABC$€000/label/license_nr/label/'+str(indexing), line)
            indexing += 1
        if re.search(r'\b07?\d{1,3}-? ?\d{3}-? ?\d{2}-? ?\d{1,2}\b', line):  # Landline number in Sweden
            line = re.sub(r'(07?\d{1,3}-? ?\d{3}-? ?\d{2}-? ?\d{1,2})', '0000-000000/label/phone_nr/label/'+str(indexing), line)
            indexing += 1
        if re.search(r'\b\d{1,2}([a-z]{2})? (\w.*) \d{2,4}\b', line):
            line = re.sub(r'\b\d{1,2}([a-z]{2})?(?= (\w.*) \d{2,4})\b', r'11/label/date_digits/label/'+str(indexing), line)
            indexing += 1
        if 'mobil' in line:  # Mobile number format
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if re.search('07\d{8}', j):
                    line_split[i] = '0000-000000/label/phone_nr/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
        else:
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if re.search(r'^\d{4}-\d{2}-\d{2}$', j):  # Date 1111-11-11
                    line_split[i] = '1111-11-11/label/date_digits/label/'+str(indexing)
                    indexing += 1
                if re.search(r'^\d{2}\/\d{2}\/\d{2}$', j):  # Date 11/11/11
                    line_split[i] = '11/11/11/label/date_digits/label/'+str(indexing)
                    indexing += 1
                if re.search(r'^\d{6}$', j) or re.search(r'^\d{8}$', j):  # Date 111111
                    line_split[i] = '111111/label/date_digits/label/'+str(indexing)
                    indexing += 1
                if re.search(r'^\d{2}\.\d{2}\.\d{2}$', j):  # Date 11.11.11
                    line_split[i] = '11.11.11/label/date_digits/label/'+str(indexing)
                    indexing += 1
                if re.search(r'^\d{1,2}/\d{2}$', j):  # Date 11/11
                    line_split[i] = '11/11/label/date_digits/label/'+str(indexing)
                    indexing += 1
                if re.search(r'^\d{4}$', j):  # Year - randomise "2018" with (-2,2) # If statement
                    line_split[i] = str(int(j) + random.randint(-2,2))+'/label/year/label/'+str(indexing)
                    indexing += 1
                # Personal number formats
                if re.search(r'\b\d{6}-\d{4}\b', j) or re.search(r'\b\d{8}-\d{4}\b', j) or re.search(r'\b\d{10}\b',j) or re.search(r'\b\d{12}\b',j):
                    line_split[i] = '123456-0000/label/personid_nr/label/'+str(indexing)
                    indexing += 1
                #if any(x!='0' for x in list(j)) and re.search('\d{10}',j):# or re.search('\d{12}', j):
                 #   line_split[i] = '123456-0000/label/personid_nr/label/'+str(indexing)
                  #  indexing += 1
                if '@' in j:  # Email addresses are formatted to email @dot.com
                    line_split[i] = 'email@dot.com/label/email/label/'+str(indexing)
                    indexing += 1
                if 'https' in j:  # https url format
                    line_split[i] = 'url.com/label/url/label/'+str(indexing)
                    indexing += 1
                if 'http' in j: # http url format
                    line_split[i] = 'url.com/label/url/label/'+str(indexing)
                    indexing += 1
                if 'www' in j:  # www web address
                    line_split[i] = 'url.com/label/url/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
            
    data = _data
    _data = []
           
    # Randomised days in the data using a list of all the days in a week
    for line in data:
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_days:
                if j == k:
                    line_split[i] = random.choice(list_days)+'/label/day/label/'+str(indexing)
                    indexing += 1
        _data.append(' '.join(line_split))
    data = _data
    _data = []
    
    # Randomised months loop
    for line in data:
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_months:
                if j == k:
                    line_split[i] = random.choice(list_months)+'/label/month_word/label/'+str(indexing)
                    indexing += 1
        _data.append(' '.join(line_split))
     
    data = _data
    _data = []
    
    for line in data:
        if re.search(r'\är snart \d{1,2} (\år)?', line): # Age mentioned in numbers are randomised with (-2,2)
            y = re.findall(r'\är snart (\d{1,2}) (\år)?', line)
            if int(y[0][0]) > 2:
                y1 = str(int(y[0][0]) + random.randint(-2,2))
                line_split = line.split(' ')
            else:
                y1 = str(int(y[0][0]) + 2)
                line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0][0]:
                    line_split[i] = y1+'/label/age_digits/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
        elif len(re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\d{1,2}) år', line)) > 0: 
            # Age mentioned in numbers are randomised with (-2,2) using RegEx
            y = re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\d{1,2}) år', line)
            if int(y[0][1]) > 2:
                y1 = str(int(y[0][1]) + random.randint(-2,2))
                line_split = line.split(' ')
            else:
                y1 = str(int(y[0][1]) + 2)
                line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0][1]:
                    line_split[i] = y1+'/label/age_digits/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
        elif re.search(r'\är snart (\w+) (\år)?', line): # Age mentioned in numbers are randomised with (-2,2)
            list_number = tuple([key for key, value in dict_numbers.items()])
            y = re.findall(r'\är snart (\w+) (\år)?', line)
            y1 = list(y[0])
            y2 = get_correct_spelling(y1[0], list_number)
            y3 = dict_numbers[y2]
            if int(y3) > 2:
                y3 = str(int(y3) + random.randint(-2,2))
                line_split = line.split(' ')
            else:
                y3 = str(int(y3) + 2)
                line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y1[0]:
                    line_split[i] = y3+'/label/age_string/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
        elif len(re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\w+) år', line)) > 0:
            # Age mentioned in words are randomised with (-2,2)
            # If the age is misspelled then it is autocorrected and then randomised
            list_number = tuple([key for key, value in dict_numbers.items()])
            y = re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\w+) år', line)
            y1 = list(y[0])
            y2 = get_correct_spelling(y1[1], list_number)
            y3 = dict_numbers[y2]
            if int(y3) > 2:
                y3 = str(int(y3) + random.randint(-2,2))
                line_split = line.split(' ')
            else:
                y3 = str(int(y3) + 2)
                line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y1[1]:
                    line_split[i] = y3+'/label/age_string/label/'+str(indexing)
                    indexing += 1
            _data.append(' '.join(line_split))
        else:
            _data.append(line)
            
    data = _data
    _data = []
    
    for line in data:
        if all(x not in line for x in ['heter','namn']):
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                for k in list_family:
                    if j == k:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_family)
                        indexing += 1
            _data.append(' '.join(line_split))
        else:
            _data.append(line)
    
    data = _data
    _data = []

    for line in data:
        if re.search(r'\t?kompisar', line):
            y = re.findall(r'([\w]+) kompisar', line)
            if not y:
                _data.append(line) 
            elif y[0] in list_siblings:
                line_split = line.split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append(' '.join(line_split))
            else:
                _data.append(line)
        elif re.search(r'\t?bröder', line):
            y = re.findall(r'([\w]+) bröder', line)
            if not y:
                _data.append(line) 
            elif y[0] in list_siblings:
                line_split = line.split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append(' '.join(line_split))
            else:
                _data.append(line) 
        elif re.search(r'\t?systern', line):
            y = re.findall(r'([\w]+) systern', line)
            if not y:
                _data.append(line) 
            elif y[0] in list_siblings:
                line_split = line.split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append(' '.join(line_split))
            else:
                _data.append(line) 
        else:
            _data.append(line)
            
    data = ' '.join(_data)
    _data = []
    
    for y,z in dict_universities.items():
        for i in z:
            if i in data:
                new_name = random.choice(list(dict_universities.keys()))
                new_univ = dict_universities[new_name][0]
                if ' ' in new_univ:
                    new_univ = new_univ.replace(' ', '$€')
                data = data.replace(i+' ', new_univ+'/label/school/label/'+str(indexing)+' ')
                indexing += 1

    data = nltk.sent_tokenize(data)
    _list = []
    _data = []
    
    list_title = list_job_title['Yrkesbenämning'].tolist()
    list_title = list_title[:-7]
    
    for x in list_title:
        if ',' in x:
            y = x.split(',')
            _list.append(y[0].lower())
        else:
            _list.append(x.lower())
            
    for line in data:
        for y in _list:
            if y in line.split(' '):
                line = line.replace(y, y+'/label/prof/label/'+str(indexing))    #random.choice(_list)
                indexing += 1
        _data.append(line)
    
    data = _data
    countries_in_data = {}
    cities_in_data = {}
    
    list_countries = list_data['Countries'].tolist()
    list_cities = list_data['Cities'].tolist()
    
    list_countries_ = list_data_['Countries'].tolist()
    list_cities_ = list_data_['Cities'].tolist()
    
    countries_nr = 1
    cities_nr = 1
    for i in data:
        for j in set(list_countries):
            if ' ' in str(j):
                if str(j) in i and list_countries.index(str(j)) not in countries_in_data:
                    countries_in_data[list_countries.index(str(j))] = (j, countries_nr)
                    countries_nr += 1
            else:
                if str(j) in i.split(' ') and list_countries.index(str(j)) not in countries_in_data:
                    countries_in_data[list_countries.index(str(j))] = (j, countries_nr)
                    countries_nr += 1
   
    for i in data:
        for k in set(list_cities):
            if ' ' in str(k):
                if str(k) in i and list_cities.index(str(k)) not in cities_in_data:
                    cities_in_data[list_cities.index(str(k))] = (k, cities_nr)
                    cities_nr += 1
            else:
                if str(k) in i.split(' ') and list_cities.index(str(k)) not in cities_in_data:
                    cities_in_data[list_cities.index(str(k))] = (k, cities_nr) 
                    cities_nr += 1
  
    _data = ' '.join(data)
    
    
    if len(countries_in_data) > 0:
        if len(cities_in_data) > 0:
            countries_occ = []
            cities_occ = []
            _city, _pos = zip(*countries_in_data.values())
            _keys = countries_in_data.keys()
            for i,j in cities_in_data.items():
                if list_countries[i] in list(_city) and list_countries[i] not in countries_occ and list_cities[i] not in cities_occ:
                    x1 = random.choice(list_countries_)
                    if ' ' in x1:
                        y1 = x1.replace(' ', '$€')
                    else:
                        y1 = x1
                    _city = list(_city)
                    _find = _city.index(list_countries[i])
                    _keys = list(_keys)
                    _find = _keys[_find]
                    _data = _data.replace(list_countries[i]+' ', y1+'/label/country/label/'+str(indexing)+' ')
                    if list_countries[i][-1] != 's':
                        if y1[-1] != 's':
                            _y1_ = y1 + 's'
                        else:
                            _y1_ = y1
                        _data = _data.replace(list_countries[i]+'s'+' ', _y1_+'/label/country/label/gen/label/'+str(indexing)+' ')
                    #+str(countries_in_data[_find][1]))
                    indexing += 1
                    countries_occ.append(list_countries[i])
                    _index = list_countries_.index(x1)
                    x2 = list_cities_[_index]
                    if ' ' in x2:
                        y2 = x2.replace(' ', '$€')
                    else:
                        y2 = x2
                    _data = _data.replace(list_cities[i]+' ', y2+'/label/city/label/'+str(indexing)+' ')#+str(j[1]))
                    if list_cities[i][-1] != 's':
                        if y2[-1] != 's':
                            _y2_ = y2 + 's'
                        else:
                            _y2_ = y2
                        _data = _data.replace(list_cities[i]+'s'+' ', _y2_+'/label/city/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    cities_occ.append(list_cities[i])
                    list_temp = [ix for ix in range(len(list_countries)) if list_countries[ix] == list_countries[i]]
                    temp_list = [xi for xi in range(len(list_countries)) if list_countries[xi] == x1]
                    for key in cities_in_data.keys():
                        if key != i and list_cities[key] not in cities_occ:
                            for jx in list_temp:
                                if list_cities[jx] == list_cities[key]:
                                    x4 = list_cities[random.choice(temp_list)]
                                    if ' ' in x4:
                                        y4 = x4.replace(' ', '$€')
                                    else:
                                        y4 = x4
                                    _data = _data.replace(list_cities[key]+' ', y4+'/label/city/label/'+str(indexing)+' ')
                                    if list_cities[key][-1] != 's':
                                        if y4[-1] != 's':
                                            _y4_ = y4 + 's'
                                        else:
                                            _y4_ = y4
                                        _data = _data.replace(list_cities[key]+'s'+' ', _y4_+'/label/city/label/gen/label/'+str(indexing)+' ')
                                    #+str(cities_in_data[key][1]))
                                    indexing += 1
                                    cities_occ.append(list_cities[key])
                elif list_cities[i] not in cities_occ:
                    x3 = random.choice(list_cities_)
                    if ' ' in x3:
                        y3 = x3.replace(' ', '$€')
                    else:
                        y3 = x3
                    _data = _data.replace(j[0]+' ', y3+'/label/city/label/'+str(indexing)+' ')#+str(j[1]))
                    if j[0][-1] != 's':
                        if y3[-1] != 's':
                            _y3_ = y3 + 's'
                        else:
                            _y3_ = y3
                        _data = _data.replace(j[0]+'s'+' ', _y3_+'/label/city/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    cities_occ.append(list_cities[i])
                               
            for i,j in countries_in_data.items():
                _street, _s_pos = zip(*cities_in_data.values())
                if list_cities[i] in list(_street) and list_cities[i] not in cities_occ and list_countries[i] not in countries_occ:
                    _x1 = random.choice(list_cities_)
                    if ' ' in _x1:
                        _y1 = _x1.replace(' ', '$€')
                    else:
                        _y1 = _x1
                    _data = _data.replace(list_cities[i]+' ', _y1+'/label/city/label/'+str(indexing)+' ')#+str(j[1]))
                    if list_cities[i][-1] != 's':
                        if _y1[-1] != 's':
                            _y1_1 = _y1 + 's'
                        else:
                            _y1_1 = _y1
                        _data = _data.replace(list_cities[i]+'s'+' ', _y1_1+'/label/city/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    cities_occ.append(list_cities[i])
                    _index = list_cities_.index(_x1)
                    _x2 = list_countries_[_index]
                    if ' ' in _x2:
                        _y2 = _x2.replace(' ', '$€')
                    else:
                        _y2 = _x2
                    _data = _data.replace(list_countries[i]+' ', _y2+'/label/country/label/'+str(indexing)+' ')#+str(j[1]))
                    if list_countries[i][-1] != 's':
                        if _y2[-1] != 's':
                            _y2_2 = _y2 + 's'
                        else:
                            _y2_2 = _y2
                        _data = _data.replace(list_countries[i]+'s'+' ', _y2_2+'/label/country/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    countries_occ.append(list_countries[i])
                elif list_countries[i] not in countries_occ:
                    _x3 = random.choice(list_countries_)
                    if ' ' in _x3:
                        _y3 = _x3.replace(' ', '$€')
                    else:
                        _y3 = _x3
                    _data = _data.replace(j[0]+' ', _y3+'/label/country/label/'+str(indexing)+' ')#+str(j[1]))
                    if j[0][-1] != 's':
                        if _y3[-1] != 's':
                            _y3_3 = _y3 + 's'
                        else:
                            _y3_3 = _y3
                        _data = _data.replace(j[0]+'s'+' ', _y3_3+'/label/country/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    countries_occ.append(list_countries[i])
        else:
            for i,j in countries_in_data.items():
                xx = random.choice(list_countries_)
                if ' ' in xx:
                    yy = xx.replace(' ', '$€')
                else:
                    yy = xx
                _data = _data.replace(j[0]+' ', yy+'/label/country/label/'+str(indexing)+' ')#+str(j[1]))
                if j[0][-1] != 's':
                    if yy[-1] != 's':
                        _yy_ = yy + 's'
                    else:
                        _yy_ = yy
                    _data = _data.replace(j[0]+'s'+' ', _yy_+'/label/country/label/gen/label/'+str(indexing)+' ')
                indexing += 1
    elif len(cities_in_data) > 0:
        for i,j in cities_in_data.items():
            xx = random.choice(list_cities_)
            if ' ' in xx:
                yy = xx.replace(' ', '$€')
            else:
                yy = xx
            _data = _data.replace(j[0]+' ', yy+'/label/city/label/'+str(indexing)+' ')#+str(j[1]))
            if j[0][-1] != 's':
                if yy[-1] != 's':
                    _yy_ = yy + 's'
                else:
                    _yy_ = yy
                _data = _data.replace(j[0]+'s'+' ', _yy_+'/label/city/label/gen/label/'+str(indexing)+' ')
            indexing += 1
            
    data = nltk.sent_tokenize(_data)  
    _data = []
    city_in_data = {}
    street_in_data = {}
    
    swe_city = swe_street_data['City'].tolist()
    swe_street = swe_street_data['Street_name'].tolist()
    
    city_nr = 1
    street_nr = 1
    for i in data:
        for j in set(swe_city):
            if ' ' in str(j):
                if str(j) in i and swe_city.index(str(j)) not in city_in_data:
                    city_in_data[swe_city.index(str(j))] = (j, city_nr)
                    city_nr += 1
            else:
                if str(j) in i.split(' ') and swe_city.index(str(j)) not in city_in_data:
                    city_in_data[swe_city.index(str(j))] = (j, city_nr)
                    city_nr += 1
   
    for i in data:
        for k in set(swe_street):
            if ' ' in str(k):
                if str(k) in i and swe_street.index(str(k)) not in street_in_data:
                    street_in_data[swe_street.index(str(k))] = (k, street_nr)
                    street_nr += 1
            else:
                if str(k) in i.split(' ') and swe_street.index(str(k)) not in street_in_data:
                    street_in_data[swe_street.index(str(k))] = (k, street_nr) 
                    street_nr += 1
  
    _data = ' '.join(data)

    new_street_data = {}
    if len(city_in_data) > 0:
        for i,j in street_in_data.items():
                x,y = zip(*city_in_data.values())
                if j[0] not in list(x):
                    new_street_data[i] = j
    else:
        new_street_data = street_in_data

    street_in_data = new_street_data

    if len(city_in_data) > 0:
        if len(street_in_data) > 0:
            stre_occ = []
            city_occ = []
            for i,j in street_in_data.items():
                _city, _pos = zip(*city_in_data.values())
                _keys = city_in_data.keys()
                if swe_city[i] in list(_city) and swe_city[i] not in city_occ and swe_street[i] not in stre_occ:
                    x1 = random.choice(swe_city)
                    if ' ' in x1:
                        y1 = x1.replace(' ', '$€')
                    else:
                        y1 = x1
                    _city = list(_city)
                    _find = _city.index(swe_city[i])
                    _keys = list(_keys)
                    _find = _keys[_find]
                    _data = _data.replace(swe_city[i]+' ', y1+'/label/city_swe/label/'+str(indexing)+' ')
                    if swe_city[i][-1] != 's':
                        if y1[-1] != 's':
                            s_y1 = y1 + 's'
                        else:
                            s_y1 = y1
                        _data = _data.replace(swe_city[i]+'s'+' ', s_y1+'/label/city_swe/label/gen/label/'+str(indexing)+' ')                         #+str(city_in_data[_find][1]))
                    indexing += 1
                    city_occ.append(swe_city[i])
                    _index = swe_city.index(x1)
                    x2 = swe_street[_index]
                    if ' ' in x2:
                        y2 = x2.replace(' ', '$€')
                    else:
                        y2 = x2
                    _data = _data.replace(swe_street[i]+' ', y2+'/label/street_nr/label/'+str(indexing)+' ')#+str(j[1]))
                    if swe_street[i][-1] != 's':
                        if y2[-1] != 's':
                            s_y2 = y2 + 's'
                        else:
                            s_y2 = y2
                        _data = _data.replace(swe_street[i]+'s'+' ', s_y2+'/label/street_nr/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    stre_occ.append(swe_street[i])
                    list_temp = [ix for ix in range(len(swe_city)) if swe_city[ix] == swe_city[i]]
                    temp_list = [xi for xi in range(len(swe_city)) if swe_city[xi] == x1]
                    for key in street_in_data.keys():
                        if key != i and swe_street[key] not in stre_occ:
                            for jx in list_temp:
                                if swe_street[jx] == swe_street[key]:
                                    x4 = swe_street[random.choice(temp_list)]
                                    if ' ' in x4:
                                        y4 = x4.replace(' ', '$€')
                                    else:
                                        y4 = x4
                                    _data = _data.replace(swe_street[key]+' ', y4+'/label/street_nr/label/'+str(indexing)+' ')
                                    if swe_street[key][-1] != 's':
                                        if y4[-1] != 's':
                                            s_y4 = y4 + 's'
                                        else:
                                            s_y4 = y4
                                        _data = _data.replace(swe_street[key]+'s'+' ', s_y4+'/label/street_nr/label/gen/label/'+str(indexing)+' ')
                                    #+str(street_in_data[key][1]))
                                    indexing += 1
                                    stre_occ.append(swe_street[key])
                elif swe_street[i] not in stre_occ:
                    x3 = random.choice(swe_street)
                    if ' ' in x3:
                        y3 = x3.replace(' ', '$€')
                    else:
                        y3 = x3
                    _data = _data.replace(j[0]+' ', y3+'/label/street_nr/label/'+str(indexing)+' ')#+str(j[1]))
                    if j[0][-1] != 's':
                        if y3[-1] != 's':
                            s_y3 = y3 + 's'
                        else:
                            s_y3 = y3
                        _data = _data.replace(j[0]+'s'+' ', s_y3+'/label/street_nr/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    stre_occ.append(swe_street[i])
                               
            for i,j in city_in_data.items():
                _street, _s_pos = zip(*street_in_data.values())
                if swe_street[i] in list(_street) and swe_street[i] not in stre_occ and swe_city[i] not in city_occ:
                    _x1 = random.choice(swe_street)
                    if ' ' in _x1:
                        _y1 = _x1.replace(' ', '$€')
                    else:
                        _y1 = _x1
                    _data = _data.replace(swe_street[i]+' ', _y1+'/label/street_nr/label/'+str(indexing)+' ')#+str(j[1]))
                    if swe_street[i][-1] != 's':
                        if _y1[-1] != 's':
                            s_y1_ = _y1 + 's'
                        else:
                            s_y1_ = _y1
                        _data = _data.replace(swe_street[i]+'s'+' ', s_y1_+'/label/street_nr/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    stre_occ.append(swe_street[i])
                    _index = swe_street.index(_x1)
                    _x2 = swe_city[_index]
                    if ' ' in _x2:
                        _y2 = _x2.replace(' ', '$€')
                    else:
                        _y2 = _x2
                    _data = _data.replace(swe_city[i]+' ', _y2+'/label/city_swe/label/'+str(indexing)+' ')#+str(j[1]))
                    if swe_city[i][-1] != 's':
                        if _y2[-1] != 's':
                            s_y2_ = _y2 + 's'
                        else:
                            s_y2_ = _y2
                        _data = _data.replace(swe_city[i]+'s'+' ', s_y2_+'/label/city_swe/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    city_occ.append(swe_city[i])
                elif swe_city[i] not in city_occ:
                    _x3 = random.choice(swe_city)
                    if ' ' in _x3:
                        _y3 = _x3.replace(' ', '$€')
                    else:
                        _y3 = _x3
                    _data = _data.replace(j[0]+' ', _y3+'/label/city_swe/label/'+str(indexing)+' ')#+str(j[1]))
                    if j[0][-1] != 's':
                        if _y3[-1] != 's':
                            s_y3_ = _y3 + 's'
                        else:
                            s_y3_ = _y3
                        _data = _data.replace(j[0]+'s'+' ', s_y3_+'/label/city_swe/label/gen/label/'+str(indexing)+' ')
                    indexing += 1
                    city_occ.append(swe_city[i])
        else:
            for i,j in city_in_data.items():
                xx = random.choice(swe_city)
                if ' ' in xx:
                    yy = xx.replace(' ', '$€')
                else:
                    yy = xx
                _data = _data.replace(j[0]+' ', yy+'/label/city_swe/label/'+str(indexing)+' ')#+str(j[1]))
                if j[0][-1] != 's':
                    if yy[-1] != 's':
                        s_yy_ = yy + 's'
                    else:
                        s_yy_ = yy
                    _data = _data.replace(j[0]+'s'+' ', s_yy_+'/label/city_swe/label/gen/label/'+str(indexing)+' ')
                indexing += 1
    elif len(street_in_data) > 0:
        for i,j in street_in_data.items():
            xx = random.choice(swe_street)
            if ' ' in xx:
                yy = xx.replace(' ', '$€')
            else:
                yy = xx
            _data = _data.replace(j[0]+' ', yy+'/label/street_nr/label/'+str(indexing)+' ')#+str(j[1]))
            if j[0][-1] != 's':
                if yy[-1] != 's':
                    s_yy_y = yy + 's'
                else:
                    s_yy_y = yy
                _data = _data.replace(j[0]+'s'+' ', s_yy_y+'/label/street_nr/label/gen/label/'+str(indexing)+' ')
            indexing += 1
            
    data = nltk.sent_tokenize(_data)
    _data = []
    list_island = list_swedish_island['Island'].tolist()

    island_in_data = {}
    island_nr = 1
    for i in data:
        for j in set(list_island):
            if ' ' in str(j):
                if str(j) in i and list_island.index(str(j)) not in island_in_data:
                    island_in_data[list_island.index(str(j))] = (j, island_nr)
                    island_nr += 1
            else:
                if str(j) in i.split(' ') and list_island.index(str(j)) not in island_in_data:
                    island_in_data[list_island.index(str(j))] = (j, island_nr)
                    island_nr += 1
                    
    _data = ' '.join(data)

    if len(island_in_data) > 0:
        for i,j in island_in_data.items():
            z1 = random.choice(list_island)
            if ' ' in z1:
                z2 = z1.replace(' ', '$€')
            else:
                z2 = z1
            _data = _data.replace(j[0]+' ', z2+'/label/island_swe/label/'+str(indexing)+' ')#+str(j[1]))
            if j[0][-1] != 's':
                if z1[-1] != 's':
                    i_z = z2 + 's'
                else:
                    i_z = z2
                _data = _data.replace(j[0]+'s'+' ', i_z+'/label/island_swe/label/gen/label/'+str(indexing)+' ')
            indexing += 1
    data = nltk.sent_tokenize(_data)
    _data = []
    
    for line in data:  # Postal Code only swedish
        if re.search(r'\b\d{3} \d{2}\b', line):
            line = re.sub(r'(\b\d{3} \d{2}\b)', '000$€00/label/zip_code/label/'+str(indexing), line)
            indexing += 1
            _data.append(line)
        else:
            _data.append(line)

    data = _data
       
    dict_tilltal_man = {}
    dict_tilltal_kvn = {}
    dict_fornamn_man = {}
    dict_fornamn_kvn = {}
    dict_neutral_namn = {}
    dict_efternamn = {}

    for line in data:
        for i in range(len(dict_names['förnamn_män'][0])):
            if dict_names['förnamn_män'][0][i][0] in line.split(' '):
                if i not in dict_fornamn_man:
                    dict_fornamn_man[i] = (len(dict_fornamn_man.keys())+1,
                                           dict_names['förnamn_män'][0][i][0],
                                           random.choice(dict_names['freq_man'][0])[0],
                                           random.choice(dict_names['freq_man'][0])[0],
                                           random.choice(dict_names['freq_man'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0]
                                          )
                else:
                    pass
        for i in range(len(dict_names['neutral_namn'][0])):
            if dict_names['neutral_namn'][0][i][0] in line.split(' '):
                if i not in dict_neutral_namn:
                    dict_neutral_namn[i] = (len(dict_neutral_namn.keys())+1,
                                           dict_names['neutral_namn'][0][i][0],
                                           random.choice(dict_names['freq_neutral'][0])[0],
                                           random.choice(dict_names['freq_neutral'][0])[0],
                                           random.choice(dict_names['freq_neutral'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0]
                                          )
                else:
                    pass
        for i in range(len(dict_names['efternamn'][0])):
            if dict_names['efternamn'][0][i][0] in line.split(' '):
                if i not in dict_efternamn:
                    dict_efternamn[i] = (len(dict_efternamn.keys())+1,
                                         dict_names['efternamn'][0][i][0],
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0]
                                        )
                else:
                    pass
        for i in range(len(dict_names['förnamn_kvinnor'][0])):
            if dict_names['förnamn_kvinnor'][0][i][0] in line.split(' '):
                if i not in dict_fornamn_kvn:
                    dict_fornamn_kvn[i] = (len(dict_fornamn_kvn.keys())+1,
                                           dict_names['förnamn_kvinnor'][0][i][0],
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0]
                                          )
                else:
                    pass


    _data = ' '.join(data)
    data = _data.split(' ')
    
    gen_male_index = {}
    gen_female_index = {}
    gen_neutral_index = {}
    
    new_data = ' '.join(data)
    for i,j in enumerate(data):                   
        for key, value in dict_fornamn_man.items():
            if value[1] == j:
                new_data = new_data.replace(data[i]+' ', str(value[2])+'/label/firstname_male/label/'+str(indexing)+' ')
                gen_male_index[value[1]] = str(indexing)
                indexing += 1

    for i,j in enumerate(data):                   
        for key, value in dict_fornamn_kvn.items():
            if value[1] == j:
                new_data = new_data.replace(data[i]+' ', str(value[2])+'/label/firstname_female/label/'+str(indexing)+' ')#+str(value[0])
                gen_female_index[value[1]] = str(indexing)
                indexing += 1

    for i,j in enumerate(data):                   
        for key, value in dict_neutral_namn.items():
            if value[1] == j:
                new_data = new_data.replace(data[i]+' ', str(value[2])+'/label/firstname_unknown/label/'+str(indexing)+' ')#+str(value[0])
                gen_neutral_index[value[1]] = str(indexing)
                indexing += 1
       
    for i,j in enumerate(data):
        for key, value in dict_efternamn.items():
            if value[1] == j:
                new_data = new_data.replace(data[i]+' ', str(value[2])+'/label/surname/label/'+str(indexing)+' ')
                indexing += 1
    
    for i,j in enumerate(data):
        for key, value in dict_fornamn_man.items():
            value_s = value[1] + 's'
            if value_s == j:
                if value[2][-1] == 's': 
                    new_data = new_data.replace(data[i], str(value[2])+'/label/firstname_male/label/gen/label/'+gen_male_index[value[1]])
                else:
                    new_data = new_data.replace(data[i], str(value[2]+'s')+'/label/firstname_male/label/gen/label/'+gen_male_index[value[1]])

    for i,j in enumerate(data):
        for key, value in dict_fornamn_kvn.items():
            value_s = value[1] + 's'
            if value_s == j:
                if value[2][-1] == 's':
                    new_data = new_data.replace(value_s, str(value[2])+'/label/firstname_female/label/gen/label/'+gen_female_index[value[1]])
                else:
                    new_data = new_data.replace(value_s, str(value[2]+'s')+'/label/firstname_female/label/gen/label/'+gen_female_index[value[1]])
    
    for i,j in enumerate(data):
        for key, value in dict_neutral_namn.items():
            value_s = value[1] + 's'
            if value_s == j:
                if value[2][-1] == 's':
                    new_data = new_data.replace(value_s, str(value[2])+'/label/firstname_unknown/label/gen/label/'+gen_neutral_index[value[1]])
                else:
                    new_data = new_data.replace(value_s, str(value[2]+'s')+'/label/firstname_unknown/label/gen/label/'+gen_neutral_index[value[1]])
    
    _data = ' '.join(data)   
    
    
    data = nltk.sent_tokenize(new_data)
    
    data_ = []
    
    for line in data:
        data_temp = []
        for token in line.split(' '):
            if '/label/' in token:
                token_list = token.split('/label/')
                if '$€' in token_list[0]:
                    new_token = ' '.join(token_list[0].split('$€'))
                else:
                    new_token = token_list[0]
                data_temp.append({'string' : new_token, 'label' : token_list[1:]})
            else:
                data_temp.append({'string' : token, 'label' : []})
        data_.append(data_temp)
     
    return data_

if __name__ == '__main__':

    output_data = identify(data)
    