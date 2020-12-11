# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
from Levenshtein import distance
import argparse, json
from os.path import abspath, dirname
from LR_project.src import sparv_annotation

MODULEDIR = dirname(dirname(abspath(__file__)))

# If it is misspelled then the function corrects the spelling and then randomly change to a digit
def get_correct_spelling(word, list_number):
    return min(list_number, key=lambda x: distance(word, x))

def is_pm_pos(annotated_data, sentenceindex, wordindex):
    #Check if a word has the part of speech Proper Noun
    #by cross-checking the index against the annotated corresponding data
    sentence = annotated_data[sentenceindex]
    word, pos = sentence.split()[wordindex].split('//')
    if pos == 'PM':
        return True
    else:
        return False

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
list_data = pd.read_csv(MODULEDIR + '/dataset/cityCountry.csv')
list_data_ = pd.read_csv(MODULEDIR + '/dataset/_cityCountry.csv')
list_swedish_cities = pd.read_csv(MODULEDIR + '/dataset/cities_sweden.csv')
list_swedish_island = pd.read_csv(MODULEDIR + '/dataset/island_sweden.csv')
swe_street_data = pd.read_csv(MODULEDIR + '/dataset/swedish_streets.csv')


# Main function to de-identify all the personal information and save as a text file
def identify(data):

    annotated_data = sparv_annotation.annotate(data)

    # To have the output format same as the input especially for newline and paragraphs in the text
    #data = re.sub(r'\n\n', ' $$$$ . ', data)  
    data = nltk.sent_tokenize(data)  # Sentence Tokenize to keep track on the

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
            source_ = re.match(r'(.*)(\b\d{4}-\d{2} \d{3} \d{2}\b)(.*)', line)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4}-\d{2} \d{3} \d{2}\b)', '0000-00$€£000$€£00/label/account_nr/label/'+str(indexing), line)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = line
            z = line
        if re.search(r'\b\d{4}-\d{3} \d{3} \d{3}\b', y):
            source_ = re.match(r'(.*)(\b\d{4}-\d{3} \d{3} \d{3}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4}-\d{3} \d{3} \d{3}\b)', '0000-000$€£000$€£000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{4}-\d{1} \d{3} \d{3} \d{4}\b', y):
            source_ = re.match(r'(.*)(\b\d{4}-\d{1} \d{3} \d{3} \d{4}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4}-\d{1} \d{3} \d{3} \d{4}\b)', '0000-0$€£000$€£000$€£0000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{4} \d{2} \d{3} \d{2}\b', y):
            source_ = re.match(r'(.*)(\b\d{4} \d{2} \d{3} \d{2}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4} \d{2} \d{3} \d{2}\b)', '0000$€£00$€£000$€£00/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{4} \d{3} \d{3} \d{3}\b', y):
            source_ = re.match(r'(.*)(\b\d{4} \d{3} \d{3} \d{3}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4} \d{3} \d{3} \d{3}\b)', '0000$€£000$€£000$€£000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{4} \d{1} \d{3} \d{3} \d{4}\b', y):
            source_ = re.match(r'(.*)(\b\d{4} \d{1} \d{3} \d{3} \d{4}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{4} \d{1} \d{3} \d{3} \d{4}\b)', '0000$€£0$€£000$€£000$€£0000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{15}\b', y):
            source_ = re.match(r'(.*)(\b\d{15}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{15}\b)', '000000000000000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{13}\b', y):
            source_ = re.match(r'(.*)(\b\d{13}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{13}\b)', '0000000000000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{11}\b', y):
            source_ = re.match(r'(.*)(\b\d{11}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{11}\b)', '10000000000/label/account_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b[A-Z]{3}\b \d{3}\b', y): # Vehicle License number
            source_ = re.match(r'(.*)(\b[A-Z]{3}\b \d{3}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b[A-Z]{3}\b \d{3}\b)', 'ABC$€£000/label/license_nr/label/' + str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b[A-Z]{3}\b \d{2}[A-Z]{1}\b', y): # Vehicle License number
            source_ = re.match(r'(.*)(\b[A-Z]{3}\b \d{2}[A-Z]{1}\b)(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b[A-Z]{3}\b \d{2}[A-Z]{1}\b)', 'ABC$€£00X/label/license_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b07?\d{1,3}-? ?\d{3}-? ?\d{2}-? ?\d{1,2}\b', y):  # Landline number in Sweden
            source_ = re.match(r'(.*)(07?\d{1,3}-? ?\d{3}-? ?\d{2}-? ?\d{1,2})(.*)', z)
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(07?\d{1,3}-? ?\d{3}-? ?\d{2}-? ?\d{1,2})', '9999-999999/label/phone_nr/label/'+str(indexing), y)
            z = source_.group(1)+x+source_.group(3)
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b\d{1,2}([a-z]{2})? (januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december) \d{2,4}\b', y):
            y = re.sub(r'\b\d{1,2}([a-z]{2})?(?= (\w.*) \d{2,4})\b', r'11/label/date_digits/label/'+str(indexing), y)
            z = z
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b(1|2)\d{3}-(0\d{1}|1(0|1|2))-(((0|1|2)\d{1})|(30|31))\b', y):  # Date 1111-11-11
            y = re.sub(r'\b(1|2)\d{3}-(0\d{1}|1(0|1|2))-(((0|1|2)\d{1})|(30|31))\b', '1111-11-11/label/date_digits/label/'+str(indexing), y)
            z = z
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b((1|2)\d{3}|\d{2})\/(0\d{1}|1(0|1|2))\/(((0|1|2)\d{1})|(30|31))\b', y):  # Date 1111/11/11
            y = re.sub(r'\b((1|2)\d{3}|\d{2})\/(0\d{1}|1(0|1|2))\/(((0|1|2)\d{1})|(30|31))\b', '9999/99/99/label/date_digits/label/'+str(indexing), y)
            z = z
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b(1|2)\d{3}\.(0\d{1}|1(0|1|2))\.(((0|1|2)\d{1})|(30|31))\b', y):  # Date 1111.11.11
            y = re.sub(r'\b(1|2)\d{3}\.(0\d{1}|1(0|1|2))\.(((0|1|2)\d{1})|(30|31))\b', '1111.11.11/label/date_digits/label/'+str(indexing), y)
            z = z
            indexing += 1
        else:
            y = y
            z = z
        if re.search(r'\b(((0|1|2)\d{1})|(30|31))\/(0\d{1}|1(0|1|2))\b', y):  # Date 11/11
            y = re.sub(r'\b(((0|1|2)\d{1})|(30|31))\/(0\d{1}|1(0|1|2))\b', '11/11/label/date_digits/label/'+str(indexing), y)
            indexing += 1
            z = z
        else:
            y = y
            z = z
        _data.append((y, z))
    
    data = _data
    _data = []
    
    for line in data:
        line_split = line[0].split(' ')
        for i, j in enumerate(line_split):
            if re.search(r'^(1|2)\d{3}$', j):  # Year - randomise "2018" with (-2,2) # If statement
                line_split[i] = str(int(j) + random.randint(-2,2))+'/label/year/label/'+str(indexing)
                indexing += 1
            # Personal number formats
            if re.search(r'\b\d{6}-\d{4}\b', j) or re.search(r'\b(1|2)\d{7}-\d{4}\b', j) or re.search(r'\b(1|2|3|4|5|6|7|8|9)\d{9}\b', j) or re.search(r'\b(1|2)\d{11}\b', j):
                line_split[i] = '123456-0000/label/personid_nr/label/' + str(indexing)
                indexing += 1
            elif re.search(r'\b\d{2}((0)\d{1}|1(0|1|2))(((0|1|2)\d{1})|(30|31))\b', j): # or re.search(r'^\d{8}$', line):  # Date 111111
                line_split[i] = '111111/label/date_digits/label/' + str(indexing)
                indexing += 1
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
        _data.append((' '.join(line_split), line[1]))
            
    data = _data
    _data = []
           
    # Randomised days in the data using a list of all the days in a week
    for line in data:
        line_split = line[0].split(' ')
        for i, j in enumerate(line_split):
            for k in list_days:
                if j == k:
                    line_split[i] = random.choice(list_days)+'/label/day/label/'+str(indexing)
                    indexing += 1
        _data.append((' '.join(line_split), line[1]))
    data = _data
    _data = []
    
    # Randomised months loop
    for line in data:
        line_split = line[0].split(' ')
        for i, j in enumerate(line_split):
            for k in list_months:
                if j == k:
                    line_split[i] = random.choice(list_months)+'/label/month_word/label/'+str(indexing)
                    indexing += 1
        _data.append((' '.join(line_split), line[1]))
     
    data = _data
    _data = []
    
    for line in data:
        if re.search(r'\är snart \d{1,2} (\år)?', line[0]): # Age mentioned in numbers are randomised with (-2,2)
            y = re.findall(r'\är snart (\d{1,2}) (\år)?', line[0])
            if int(y[0][0]) > 2:
                y1 = str(int(y[0][0]) + random.randint(-2,2))
                line_split = line[0].split(' ')
            else:
                y1 = str(int(y[0][0]) + 2)
                line_split = line[0].split(' ')
            for i, j in enumerate(line_split):
                if j == y[0][0]:
                    line_split[i] = y1+'/label/age_digits/label/'+str(indexing)
                    indexing += 1
            _data.append((' '.join(line_split), line[1]))
        elif len(re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\d{1,2}) (snart|år)', line[0])) > 0: 
            # Age mentioned in numbers are randomised with (-2,2) using RegEx
            y = re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\d{1,2}) (snart|år)', line[0])
            if int(y[0][1]) > 2:
                y1 = str(int(y[0][1]) + random.randint(-2,2))
                line_split = line[0].split(' ')
            else:
                y1 = str(int(y[0][1]) + 2)
                line_split = line[0].split(' ')
            for i, j in enumerate(line_split):
                if j == y[0][1]:
                    line_split[i] = y1+'/label/age_digits/label/'+str(indexing)
                    indexing += 1
            _data.append((' '.join(line_split), line[1]))
        elif re.search(r'\är snart (\w+) (\år)', line[0]): # Age mentioned in string are randomised with (-2,2)
            list_number = tuple([key for key, value in dict_numbers.items()])
            y = re.findall(r'\är snart (\w+) (\år)', line)
            y1 = list(y[0])
            y2 = get_correct_spelling(y1[0], list_number)
            y3 = dict_numbers[y2]
            if int(y3) > 2:
                y3 = str(int(y3) + random.randint(-2,2))
                line_split = line[0].split(' ')
            else:
                y3 = str(int(y3) + 2)
                line_split = line[0].split(' ')
            for i, j in enumerate(line_split):
                if j == y1[0]:
                    line_split[i] = y3+'/label/age_string/label/'+str(indexing)
                    indexing += 1
            _data.append((' '.join(line_split), line[1]))
        elif len(re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\w+) (snart|år)', line[0])) > 0:
            # Age mentioned in words are randomised with (-2,2)
            # If the age is misspelled then it is autocorrected and then randomised
            list_number = tuple([key for key, value in dict_numbers.items()])
            y = re.findall(r'(\är|fylla|fyller|fyllde|fyllt) (\w+) (snart|år)', line[0])
            y1 = list(y[0])
            y2 = get_correct_spelling(y1[1], list_number)
            y3 = dict_numbers[y2]
            if int(y3) > 2:
                y3 = str(int(y3) + random.randint(-2,2))
                line_split = line[0].split(' ')
            else:
                y3 = str(int(y3) + 2)
                line_split = line[0].split(' ')
            for i, j in enumerate(line_split):
                if j == y1[1]:
                    line_split[i] = y3+'/label/age_string/label/'+str(indexing)
                    indexing += 1
            _data.append((' '.join(line_split), line[1]))
        else:
            _data.append((line[0], line[1]))
            
    data = _data
    _data = []
    
    for line in data:
        if all(x not in line[0] for x in ['heter','namn']):
            line_split = line[0].split(' ')
            for i, j in enumerate(line_split):
                for k in list_family:
                    if j == k:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_family)
                        indexing += 1
            _data.append((' '.join(line_split), line[1]))
        else:
            _data.append((line[0], line[1]))
    
    data = _data
    _data = []

    for line in data:
        if re.search(r'\t?kompisar', line[0]):
            y = re.findall(r'([\w]+) kompisar', line[0])
            if not y:
                _data.append((line[0], line[1])) 
            elif y[0] in list_siblings:
                line_split = line[0].split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append((' '.join(line_split), line[1]))
        else:
            _data.append((line[0], line[1]))
    data = _data
    _data = []

    for line in data:
        if re.search(r'\t?bröder', line[0]):
            y = re.findall(r'([\w]+) bröder', line[0])
            if not y:
                _data.append(line[0], line[1]) 
            elif y[0] in list_siblings:
                line_split = line[0].split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append((' '.join(line_split), line[1]))
        else:
            _data.append((line[0], line[1]))
    data = _data
    _data = []

    for line in data:
        if re.search(r'\t?systrar', line[0]):
            y = re.findall(r'([\w]+) systrar', line[0])
            if not y:
                _data.append((line[0], line[1])) 
            elif y[0] in list_siblings:
                line_split = line[0].split(' ')
                for i, j in enumerate(line_split):
                    if j == y[0]:
                        line_split[i] = j+'/label/sensitive/label/'+str(indexing)    #random.choice(list_siblings)
                        indexing += 1
                _data.append((' '.join(line_split), line[1]))
        else:
            _data.append((line[0], line[1]))

    _1, _2 = zip(*_data)
    _1 = ' '.join(list(_1))
    _2 = ' '.join(list(_2))    
    data = [_1, _2]
    _data = []
    
    for y,z in dict_universities.items():
        for i in z:
            if i in data[0]:
                new_name = random.choice(list(dict_universities.keys()))
                new_univ = dict_universities[new_name][0]
                if ' ' in new_univ:
                    new_univ = new_univ.replace(' ', '$€£')
                j = i.replace(' ', '$€£')
                data[0] = data[0].replace(i+' ', new_univ+'/label/school/label/'+str(indexing)+' ')
                data[1] = data[1].replace(i+' ', j+'/label/school/label/'+str(indexing)+' ')
                indexing += 1

    _1 = nltk.sent_tokenize(data[0])
    _2 = nltk.sent_tokenize(data[1])
    data = [(_1[i],_2[i]) for i in range(len(_1))]
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
        x = ''
        for y in _list:
            if y in line[0].split(' '):
                x = line[0].replace(y, y+'/label/prof/label/'+str(indexing))    #random.choice(_list)
                indexing += 1
        if not x:
            _data.append((line[0], line[1]))
        else:
            _data.append((x,line[1]))
    
    data = _data
    countries_in_data = {}
    cities_in_data = {}
    
    # Countries and cities in the world
    list_countries = list_data['Countries'].tolist()
    list_cities = list_data['Cities'].tolist()
    
    # List "country -> city" most popular based on cities
    list_countries_ = list_data_['Countries'].tolist() 
    list_cities_ = list_data_['Cities'].tolist()

    _list_countries, _list_cities = [], []
    for i in range(len(list_countries)):
        if type(list_countries[i]) is float or type(list_cities[i]) is float:
            pass
        else:
            _list_countries.append(list_countries[i])
            _list_cities.append(list_cities[i])

    list_countries = _list_countries
    list_cities = _list_cities

    city_country_zip = zip(list_countries, list_cities)
    city_country_zip = sorted(city_country_zip, key=lambda x: x[1].count(' '), reverse=True)
    list_countries, list_cities = zip(*city_country_zip)

    countries_nr = 1
    cities_nr = 1
    _add_cou, _add_city = [], []
    line_index = 0
    for i in data:
        for j in set(list_countries):
            if ' ' in str(j):
                if str(j) in i[0] and list_countries.index(str(j)) not in countries_in_data:
                    countries_in_data[list_countries.index(str(j))] = (j, countries_nr)
                    countries_nr += 1
                    _add_cou.append(j)
            else:
                if str(j) in i[0].split(' '):
                    found_name = str(j)
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)
                
                    if list_countries.index(str(j)) not in countries_in_data and is_pm:
                        countries_in_data[list_countries.index(str(j))] = (j, countries_nr)
                        countries_nr += 1
                elif str(j).lower() in i[0].split(' '):
                    found_name = str(j).lower()
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)
                
                    if list_countries.index(str(j)) not in countries_in_data and is_pm:
                        countries_in_data[list_countries.index(str(j))] = (j.lower(), countries_nr)
                        countries_nr += 1
        line_index += 1
   
    line_index = 0
    for i in data:
        for k in set(list_cities):
            if ' ' in str(k):
                if str(k) in i[0] and list_cities.index(str(k)) not in cities_in_data:
                    cities_in_data[list_cities.index(str(k))] = (k, cities_nr)
                    cities_nr += 1
                    _add_city.append(k)
            else:
                if str(k) in i[0].split(' '):
                    found_name = str(k)
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if list_cities.index(str(k)) not in cities_in_data and is_pm:
                        cities_in_data[list_cities.index(str(k))] = (k, cities_nr)  
                        cities_nr += 1
                elif str(k).lower() in i[0].split(' '):
                    found_name = str(k).lower()
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if list_cities.index(str(k)) not in cities_in_data and is_pm:
                        cities_in_data[list_cities.index(str(k))] = (k.lower(), cities_nr) 
                        cities_nr += 1
        line_index += 1

    _1, _2 = zip(*data)
    _1 = ' '.join(list(_1))
    _2 = ' '.join(list(_2))
    for i in _add_city:
        if i in _1:
            j = i.replace(' ', '$€£')
            _1 = _1.replace(i, j)
            _2 = _2.replace(i, j)
    for i in _add_cou:
        if i in _1:
            j = i.replace(' ', '$€£')
            _1 = _1.replace(i, j)
            _2 = _2.replace(i, j)

    _1 = '?0=£€$'.join(_1.split(' '))
    _2 = '?0=£€$'.join(_2.split(' '))
    _data = [_1, _2]

    if len(countries_in_data) > 0:
        if len(cities_in_data) > 0:
            countries_occ = []
            cities_occ = []
            _country, _pos = zip(*countries_in_data.values())
            _keys = countries_in_data.keys()
            for i,j in cities_in_data.items():
                if list_countries[i] in list(_country) and list_countries[i] not in countries_occ and list_cities[i] not in cities_occ:
                    x1 = random.choice(list_countries_)
                    if ' ' in x1:
                        y1 = x1.replace(' ', '$€£')
                    else:
                        y1 = x1
                    _country = list(_country)
                    _find = _country.index(list_countries[i])
                    _keys = list(_keys)
                    _find = _keys[_find]
                    cou_modify = '$€£'.join(list_countries[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+cou_modify+'?0=£€$', '?0=£€$'+y1+'/label/country/label/'+str(indexing)+'?0=£€$')
                    if list_countries[i][-1] != 's':
                        if y1[-1] != 's':
                            _y1_ = y1 + 's'
                        else:
                            _y1_ = y1
                        _data[0] = _data[0].replace('?0=£€$'+cou_modify+'s', '?0=£€$'+_y1_+'/label/country/label/gen/label/'+str(indexing))
                    indexing += 1
                    countries_occ.append(list_countries[i])
                    _index = list_countries_.index(x1)
                    x2 = list_cities_[_index]
                    if ' ' in x2:
                        y2 = x2.replace(' ', '$€£')
                    else:
                        y2 = x2
                    cit_modify = '$€£'.join(list_cities[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+cit_modify+'?0=£€$', '?0=£€$'+y2+'/label/city/label/foreign/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if list_cities[i][-1] != 's':
                        if y2[-1] != 's':
                            _y2_ = y2 + 's'
                        else:
                            _y2_ = y2
                        _data[0] = _data[0].replace('?0=£€$'+cit_modify+'s', '?0=£€$'+_y2_+'/label/city/label/foreign/label/gen/label/'+str(indexing))
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
                                        y4 = x4.replace(' ', '$€£')
                                    else:
                                        y4 = x4
                                    _cit_modify = '$€£'.join(list_cities[key].split(' '))
                                    _data[0] = _data[0].replace('?0=£€$'+_cit_modify+'?0=£€$', '?0=£€$'+y4+'/label/city/label/foreign/label/'+str(indexing)+'?0=£€$')
                                    if list_cities[key][-1] != 's':
                                        if y4[-1] != 's':
                                            _y4_ = y4 + 's'
                                        else:
                                            _y4_ = y4
                                        _data[0] = _data[0].replace('?0=£€$'+_cit_modify+'s', '?0=£€$'+_y4_+'/label/city/label/foreign/label/gen/label/'+str(indexing))
                                    indexing += 1
                                    cities_occ.append(list_cities[key])
                elif list_cities[i] not in cities_occ:
                    x3 = random.choice(list_cities_)
                    if ' ' in x3:
                        y3 = x3.replace(' ', '$€£')
                    else:
                        y3 = x3
                    _1_cit_modify = '$€£'.join(j[0].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_1_cit_modify+'?0=£€$', '?0=£€$'+y3+'/label/city/label/foreign/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if j[0][-1] != 's':
                        if y3[-1] != 's':
                            _y3_ = y3 + 's'
                        else:
                            _y3_ = y3
                        _data[0] = _data[0].replace('?0=£€$'+_1_cit_modify+'s', '?0=£€$'+_y3_+'/label/city/label/foreign/label/gen/label/'+str(indexing))
                    indexing += 1
                    cities_occ.append(list_cities[i])
                               
            for i,j in countries_in_data.items():
                _street, _s_pos = zip(*cities_in_data.values())
                if list_cities[i] in list(_street) and list_cities[i] not in cities_occ and list_countries[i] not in countries_occ:
                    _x1 = random.choice(list_cities_)
                    if ' ' in _x1:
                        _y1 = _x1.replace(' ', '$€£')
                    else:
                        _y1 = _x1
                    _2_cit_modify = '$€£'.join(list_cities[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_2_cit_modify+'?0=£€$', '?0=£€$'+_y1+'/label/city/label/foreign/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if list_cities[i][-1] != 's':
                        if _y1[-1] != 's':
                            _y1_1 = _y1 + 's'
                        else:
                            _y1_1 = _y1
                        _data[0] = _data[0].replace('?0=£€$'+_2_cit_modify+'s', '?0=£€$'+_y1_1+'/label/city/label/foreign/label/gen/label/'+str(indexing))
                    indexing += 1
                    cities_occ.append(list_cities[i])
                    _index = list_cities_.index(_x1)
                    _x2 = list_countries_[_index]
                    if ' ' in _x2:
                        _y2 = _x2.replace(' ', '$€£')
                    else:
                        _y2 = _x2
                    _cou_modify = '$€£'.join(list_countries[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_cou_modify+'?0=£€$', '?0=£€$'+_y2+'/label/country/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if list_countries[i][-1] != 's':
                        if _y2[-1] != 's':
                            _y2_2 = _y2 + 's'
                        else:
                            _y2_2 = _y2
                        _data[0] = _data[0].replace('?0=£€$'+_cou_modify+'s', '?0=£€$'+_y2_2+'/label/country/label/gen/label/'+str(indexing))
                    indexing += 1
                    countries_occ.append(list_countries[i])
                elif list_countries[i] not in countries_occ:
                    _x3 = random.choice(list_countries_)
                    if ' ' in _x3:
                        _y3 = _x3.replace(' ', '$€£')
                    else:
                        _y3 = _x3
                    _1_cou_modify = '$€£'.join(j[0].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_1_cou_modify+'?0=£€$', '?0=£€$'+_y3+'/label/country/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if j[0][-1] != 's':
                        if _y3[-1] != 's':
                            _y3_3 = _y3 + 's'
                        else:
                            _y3_3 = _y3
                        _data[0] = _data[0].replace('?0=£€$'+_1_cou_modify+'s', '?0=£€$'+_y3_3+'/label/country/label/gen/label/'+str(indexing))
                    indexing += 1
                    countries_occ.append(list_countries[i])
        else:
            for i, j in countries_in_data.items():
                xx = random.choice(list_countries_)
                if ' ' in xx:
                    yy = xx.replace(' ', '$€£')
                else:
                    yy = xx
                _2_cou_modify = '$€£'.join(j[0].split(' '))
                _data[0] = _data[0].replace('?0=£€$'+_2_cou_modify+'?0=£€$', '?0=£€$'+yy+'/label/country/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                if j[0][-1] != 's':
                    if yy[-1] != 's':
                        _yy_ = yy + 's'
                    else:
                        _yy_ = yy
                    _data[0] = _data[0].replace('?0=£€$'+_2_cou_modify+'s', '?0=£€$'+_yy_+'/label/country/label/gen/label/'+str(indexing))
                indexing += 1
    elif len(cities_in_data) > 0:
        for i,j in cities_in_data.items():
            xx = random.choice(list_cities_)
            if ' ' in xx:
                yy = xx.replace(' ', '$€£')
            else:
                yy = xx
            _3_cit_modify = '$€£'.join(j[0].split(' '))
            _data[0] = _data[0].replace('?0=£€$'+_3_cit_modify+'?0=£€$', '?0=£€$'+yy+'/label/city/label/foreign/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
            if j[0][-1] != 's':
                if yy[-1] != 's':
                    _yy_ = yy + 's'
                else:
                    _yy_ = yy
                _data[0] = _data[0].replace('?0=£€$'+_3_cit_modify+'s', '?0=£€$'+_yy_+'/label/city/label/foreign/label/gen/label/'+str(indexing))
            indexing += 1

    _1 = ' '.join(_data[0].split('?0=£€$'))
    _2 = ' '.join(_data[1].split('?0=£€$'))
    _1 = nltk.sent_tokenize(_1)  
    _2 = nltk.sent_tokenize(_2)
    data = [(_1[i],_2[i]) for i in range(len(_1))]
    _data = []
    city_in_data = {}
    street_in_data = {}
    
    swe_city = swe_street_data['City'].tolist()
    swe_street = swe_street_data['Street_name'].tolist()

    swe_city_street = zip(swe_city, swe_street)
    swe_city_street = sorted(swe_city_street, key=lambda x: x[1].count(' '), reverse=True)
    swe_city, swe_street = zip(*swe_city_street)

    
    city_nr = 1
    street_nr = 1
    _add_city_swe, _add_street_swe = [], []
    line_index = 0
    for i in data:
        for j in set(swe_city):
            if ' ' in str(j):
                if str(j) in i[0] and swe_city.index(str(j)) not in city_in_data:
                    city_in_data[swe_city.index(str(j))] = (j, city_nr)
                    city_nr += 1
                    _add_city_swe.append(j)
            else:
                if str(j) in i[0].split(' '):
                    found_name = str(j)
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if swe_city.index(str(j)) not in city_in_data and is_pm:
                        city_in_data[swe_city.index(str(j))] = (j, city_nr)
                        city_nr += 1
                elif str(j).lower() in i[0].split(' '):
                    found_name = str(j).lower()
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if swe_city.index(str(j)) not in city_in_data and is_pm:
                        city_in_data[swe_city.index(str(j))] = (j.lower(), city_nr)
                        city_nr += 1
        line_index += 1
   
    line_index = 0
    for i in data:
        for k in set(swe_street):
            if ' ' in str(k):
                if str(k) in i[0] and swe_street.index(str(k)) not in street_in_data:
                    street_in_data[swe_street.index(str(k))] = (k, street_nr)
                    street_nr += 1
                    _add_street_swe.append(k)
            else:
                if str(k) in i[0].split(' '):
                    found_name = str(k)
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if swe_street.index(str(k)) not in street_in_data and is_pm:
                        street_in_data[swe_street.index(str(k))] = (k, street_nr) 
                        street_nr += 1
                elif str(k).lower() in i[0].split(' '):
                    found_name = str(k).lower()
                    name_index = i[0].split(' ').index(found_name)
                    is_pm = is_pm_pos(annotated_data, line_index, name_index)

                    if swe_street.index(str(k)) not in street_in_data and is_pm:
                        street_in_data[swe_street.index(str(k))] = (k.lower(), street_nr) 
                        street_nr += 1
        line_index += 1
  
    _1, _2 = zip(*data)
    _1 = ' '.join(list(_1))
    _2 = ' '.join(list(_2))
    for i in _add_city_swe:
        if i in _1:
            j = i.replace(' ', '$€£')
            _1 = _1.replace(i, j)
            _2 = _2.replace(i, j)
    for i in _add_street_swe:
        if i in _1:
            j = i.replace(' ', '$€£')
            _1 = _1.replace(i, j)
            _2 = _2.replace(i, j)

    _1 = '?0=£€$'.join(_1.split(' '))
    _2 = '?0=£€$'.join(_2.split(' '))
    _data = [_1, _2]

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
                        y1 = x1.replace(' ', '$€£')
                    else:
                        y1 = x1
                    _city = list(_city)
                    _find = _city.index(swe_city[i])
                    _keys = list(_keys)
                    _find = _keys[_find]
                    _1_swe_city = '$€£'.join(swe_city[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_1_swe_city+'?0=£€$', '?0=£€$'+y1+'/label/city/label/'+str(indexing)+'?0=£€$')
                    if swe_city[i][-1] != 's':
                        if y1[-1] != 's':
                            s_y1 = y1 + 's'
                        else:
                            s_y1 = y1
                        _data[0] = _data[0].replace('?0=£€$'+_1_swe_city+'s', '?0=£€$'+s_y1+'/label/city/label/gen/label/'+str(indexing))                         #+str(city_in_data[_find][1]))
                    indexing += 1
                    city_occ.append(swe_city[i])
                    _index = swe_city.index(x1)
                    x2 = swe_street[_index]
                    if ' ' in x2:
                        y2 = x2.replace(' ', '$€£')
                    else:
                        y2 = x2
                    _1_swe_street = '?$€£'.join(swe_street[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_1_swe_street+'?0=£€$', '?0=£€$'+y2+'/label/place/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if swe_street[i][-1] != 's':
                        if y2[-1] != 's':
                            s_y2 = y2 + 's'
                        else:
                            s_y2 = y2
                        _data[0] = _data[0].replace('?0=£€$'+_1_swe_street+'s', '?0=£€$'+s_y2+'/label/place/label/gen/label/'+str(indexing))
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
                                        y4 = x4.replace(' ', '$€£')
                                    else:
                                        y4 = x4
                                    _2_swe_street = '$€£'.join(swe_street[key].split(' '))
                                    _data[0] = _data[0].replace('?0=£€$'+_2_swe_street+'?0=£€$', '?0=£€$'+y4+'/label/place/label/'+str(indexing)+'?0=£€$')
                                    if swe_street[key][-1] != 's':
                                        if y4[-1] != 's':
                                            s_y4 = y4 + 's'
                                        else:
                                            s_y4 = y4
                                        _data[0] = _data[0].replace('?0=£€$'+_2_swe_street+'s', '?0=£€$'+s_y4+'/label/place/label/gen/label/'+str(indexing))
                                    indexing += 1
                                    stre_occ.append(swe_street[key])
                elif swe_street[i] not in stre_occ:
                    x3 = random.choice(swe_street)
                    if ' ' in x3:
                        y3 = x3.replace(' ', '$€£')
                    else:
                        y3 = x3
                    _3_swe_street = '$€£'.join(j[0].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_3_swe_street+'?0=£€$', '?0=£€$'+y3+'/label/place/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if j[0][-1] != 's':
                        if y3[-1] != 's':
                            s_y3 = y3 + 's'
                        else:
                            s_y3 = y3
                        _data[0] = _data[0].replace('?0=£€$'+_3_swe_street+'s', '?0=£€$'+s_y3+'/label/place/label/gen/label/'+str(indexing))
                    indexing += 1
                    stre_occ.append(swe_street[i])
                               
            for i,j in city_in_data.items():
                _street, _s_pos = zip(*street_in_data.values())
                if swe_street[i] in list(_street) and swe_street[i] not in stre_occ and swe_city[i] not in city_occ:
                    _x1 = random.choice(swe_street)
                    if ' ' in _x1:
                        _y1 = _x1.replace(' ', '$€£')
                    else:
                        _y1 = _x1
                    _4_swe_street = '$€£'.join(swe_street[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_4_swe_street+'?0=£€$', '?0=£€$'+_y1+'/label/place/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if swe_street[i][-1] != 's':
                        if _y1[-1] != 's':
                            s_y1_ = _y1 + 's'
                        else:
                            s_y1_ = _y1
                        _data[0] = _data[0].replace('?0=£€$'+_4_swe_street+'s','?0=£€$'+s_y1_+'/label/place/label/gen/label/'+str(indexing))
                    indexing += 1
                    stre_occ.append(swe_street[i])
                    _index = swe_street.index(_x1)
                    _x2 = swe_city[_index]
                    if ' ' in _x2:
                        _y2 = _x2.replace(' ', '$€£')
                    else:
                        _y2 = _x2
                    _2_swe_city = '$€£'.join(swe_city[i].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_2_swe_city+'?0=£€$', '?0=£€$'+_y2+'/label/city/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if swe_city[i][-1] != 's':
                        if _y2[-1] != 's':
                            s_y2_ = _y2 + 's'
                        else:
                            s_y2_ = _y2
                        _data[0] = _data[0].replace('?0=£€$'+_2_swe_city+'s', '?0=£€$'+s_y2_+'/label/city/label/gen/label/'+str(indexing))
                    indexing += 1
                    city_occ.append(swe_city[i])
                elif swe_city[i] not in city_occ:
                    _x3 = random.choice(swe_city)
                    if ' ' in _x3:
                        _y3 = _x3.replace(' ', '$€£')
                    else:
                        _y3 = _x3
                    _3_swe_city = '$€£'.join(j[0].split(' '))
                    _data[0] = _data[0].replace('?0=£€$'+_3_swe_city+'?0=£€$', '?0=£€$'+_y3+'/label/city/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                    if j[0][-1] != 's':
                        if _y3[-1] != 's':
                            s_y3_ = _y3 + 's'
                        else:
                            s_y3_ = _y3
                        _data[0] = _data[0].replace('?0=£€$'+_3_swe_city+'s', '?0=£€$'+s_y3_+'/label/city/label/gen/label/'+str(indexing))
                    indexing += 1
                    city_occ.append(swe_city[i])
        else:
            for i,j in city_in_data.items():
                xx = random.choice(swe_city)
                if ' ' in xx:
                    yy = xx.replace(' ', '$€£')
                else:
                    yy = xx
                _4_swe_city = '$€£'.join(j[0].split(' '))
                _data[0] = _data[0].replace('?0=£€$'+_4_swe_city+'?0=£€$', '?0=£€$'+yy+'/label/city/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
                if j[0][-1] != 's':
                    if yy[-1] != 's':
                        s_yy_ = yy + 's'
                    else:
                        s_yy_ = yy
                    _data[0] = _data[0].replace('?0=£€$'+_4_swe_city+'s', '?0=£€$'+s_yy_+'/label/city/label/gen/label/'+str(indexing))
                indexing += 1
    elif len(street_in_data) > 0:
        for i,j in street_in_data.items():
            xx = random.choice(swe_street)
            if ' ' in xx:
                yy = xx.replace(' ', '$€£')
            else:
                yy = xx
            _5_swe_street = '$€£'.join(j[0].split(' '))
            _data[0] = _data[0].replace('?0=£€$'+_5_swe_street+'?0=£€$', '?0=£€$'+yy+'/label/place/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
            if j[0][-1] != 's':
                if yy[-1] != 's':
                    s_yy_y = yy + 's'
                else:
                    s_yy_y = yy
                _data[0] = _data[0].replace('?0=£€$'+_5_swe_street+'s', '?0=£€$'+s_yy_y+'/label/place/label/gen/label/'+str(indexing))
            indexing += 1

    _1 = ' '.join(_data[0].split('?0=£€$'))
    _2 = ' '.join(_data[1].split('?0=£€$'))
    _1 = nltk.sent_tokenize(_1)  
    _2 = nltk.sent_tokenize(_2)
    data = [(_1[i],_2[i]) for i in range(len(_1))]

    _data = []
    list_island = list_swedish_island['Island'].tolist()

    island_in_data = {}
    island_nr = 1
    _add_island = []
    for i in data:
        for j in set(list_island):
            if ' ' in str(j):
                if str(j) in i[0] and list_island.index(str(j)) not in island_in_data:
                    island_in_data[list_island.index(str(j))] = (j, island_nr)
                    island_nr += 1
                    _add_island.append(j)
            else:
                if str(j) in i[0].split(' ') and list_island.index(str(j)) not in island_in_data:
                    island_in_data[list_island.index(str(j))] = (j, island_nr)
                    island_nr += 1

    _1, _2 = zip(*data)
    _1 = ' '.join(list(_1))
    _2 = ' '.join(list(_2))
    for i in _add_island:
        if i in _1:
            j = i.replace(' ', '$€£')
            _1 = _1.replace(i, j)
            _2 = _2.replace(i, j)
    
    _1 = '?0=£€$'.join(_1.split(' '))
    _2 = '?0=£€$'.join(_2.split(' '))

    _data = [_1, _2]                

    if len(island_in_data) > 0:
        for i,j in island_in_data.items():
            z1 = random.choice(list_island)
            if ' ' in z1:
                z2 = z1.replace(' ', '$€£')
            else:
                z2 = z1
            _2_iis = '$€£'.join(j[0].split(' '))
            _data[0] = _data[0].replace('?0=£€$'+_2_iis+'?0=£€$', '?0=£€$'+z2+'/label/island/label/'+str(indexing)+'?0=£€$')#+str(j[1]))
            if j[0][-1] != 's':
                if z1[-1] != 's':
                    i_z = z2 + 's'
                else:
                    i_z = z2
                _data[0] = _data[0].replace('?0=£€$'+_2_iis+'s', '?0=£€$'+i_z+'/label/island/label/gen/label/'+str(indexing))
            indexing += 1

    _1 = ' '.join(_data[0].split('?0=£€$'))
    _2 = ' '.join(_data[1].split('?0=£€$'))
    _1 = nltk.sent_tokenize(_1)  
    _2 = nltk.sent_tokenize(_2)
    data = [(_1[i],_2[i]) for i in range(len(_1))]      
    _data = []
    
    for line in data:  # Postal Code only swedish
        if re.search(r'\b\d{3} ?\d{2}\b', line[0]):
            source_ = re.match(r'(.*)(\b\d{3} ?\d{2}\b)(.*)', line[1])
            x = source_.group(2).replace(' ', '$€£')
            y = re.sub(r'(\b\d{3} ?\d{2}\b)', '000$€£00/label/zip_code/label/'+str(indexing), line[0])
            indexing += 1
            _data.append((y, source_.group(1)+x+source_.group(3)))
        else:
            _data.append((line[0], line[1]))

    data = _data
       
    dict_tilltal_man = {}
    dict_tilltal_kvn = {}
    dict_fornamn_man = {}
    dict_fornamn_kvn = {}
    dict_neutral_namn = {}
    dict_efternamn = {}

    line_index = 0
    for line in data:
        for i in range(len(dict_names['förnamn_män'][0])):
            if dict_names['förnamn_män'][0][i][0] in line[0].split(' '):
                found_name = dict_names['förnamn_män'][0][i][0]
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun

                if i not in dict_fornamn_man and is_pm:
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
            elif dict_names['förnamn_män'][0][i][0].lower() in line[0].split(' '):
                found_name = dict_names['förnamn_män'][0][i][0].lower()
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
            
                if i not in dict_fornamn_man and is_pm:
                    dict_fornamn_man[i] = (len(dict_fornamn_man.keys())+1,
                                           dict_names['förnamn_män'][0][i][0].lower(),
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
            if dict_names['neutral_namn'][0][i][0] in line[0].split(' '):
                found_name = dict_names['neutral_namn'][0][i][0]
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                
                if i not in dict_neutral_namn and is_pm:
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
            elif dict_names['neutral_namn'][0][i][0].lower() in line[0].split(' '):
                found_name = dict_names['neutral_namn'][0][i][0].lower()
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                
                if i not in dict_neutral_namn and is_pm:
                    dict_neutral_namn[i] = (len(dict_neutral_namn.keys())+1,
                                           dict_names['neutral_namn'][0][i][0].lower(),
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
            if dict_names['efternamn'][0][i][0] in line[0].split(' '):
                found_name = dict_names['efternamn'][0][i][0]
                name_index = line[0].split(' ').index(found_name)  #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                
                if i not in dict_efternamn and is_pm:
                    dict_efternamn[i] = (len(dict_efternamn.keys())+1,
                                         dict_names['efternamn'][0][i][0],
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0]
                                        )
                else:
                    pass
            elif dict_names['efternamn'][0][i][0].lower() in line[0].split(' '):
                found_name = dict_names['efternamn'][0][i][0].lower()
                name_index = line[0].split(' ').index(found_name)  #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                if i not in dict_efternamn and is_pm:
                    dict_efternamn[i] = (len(dict_efternamn.keys())+1,
                                         dict_names['efternamn'][0][i][0].lower(),
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0],
                                         random.choice(dict_names['freq_efternamn'][0])[0]
                                        )
                else:
                    pass
        for i in range(len(dict_names['förnamn_kvinnor'][0])):
            if dict_names['förnamn_kvinnor'][0][i][0] in line[0].split(' '):
                found_name = dict_names['förnamn_kvinnor'][0][i][0]
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                
                if i not in dict_fornamn_kvn and is_pm:
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
            elif dict_names['förnamn_kvinnor'][0][i][0].lower() in line[0].split(' '):
                found_name = dict_names['förnamn_kvinnor'][0][i][0].lower()
                name_index = line[0].split(' ').index(found_name) #find the name as index in line
                is_pm = is_pm_pos(annotated_data, line_index, name_index) #check if the corresponding word on the corresponding line in annotated_data is a personal noun
                
                if i not in dict_fornamn_kvn and is_pm:
                    dict_fornamn_kvn[i] = (len(dict_fornamn_kvn.keys())+1,
                                           dict_names['förnamn_kvinnor'][0][i][0].lower(),
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_kvn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0],
                                           random.choice(dict_names['freq_efternamn'][0])[0]
                                          )
                else:
                    pass
        line_index += 1


    _1, _2 = zip(*data)
    _1 = ' '.join(list(_1))
    _2 = ' '.join(list(_2))
    data = _1.split(' ')
    
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
    
    _1 = ' '.join(data)   
    
    
    new_data = new_data.replace('9999/99/99', '1111/11/11')
    new_data = new_data.replace('9999-999999', '0000-000000')
    _3 = nltk.sent_tokenize(new_data)
    _2 = nltk.sent_tokenize(_2)
    data = [(_3[i], _2[i]) for i in range(len(_3))]
    data_ = []
    data_1 = []
    for line in data:
        data_temp = []
        s_1 = line[0].split(' ')
        s_2 = line[1].split(' ')
        for token in range(len(s_1)):
            if '/label/' in s_1[token]:
                token_list = s_1[token].split('/label/')
                if '$€£' in token_list[0]:
                    new_token = ' '.join(token_list[0].split('$€£'))
                else:
                    new_token = token_list[0]
                data_temp.append({'source' : ' '.join(s_2[token].split('$€£')), 'target' : new_token, 'label' : token_list[1:]})
            else:
                data_temp.append({'source' : ' '.join(s_2[token].split('$€£')), 'target' : s_1[token], 'label' : []})
        data_.append(data_temp)

        data_temp_1 = []
        s_1_1 = line[0].split(' ')
        s_2_1 = line[1].split(' ')
        for token in range(len(s_1_1)):
            if '/label/' in s_1_1[token]:
                token_list = s_1_1[token].split('/label/')
                if '$€£' in token_list[0]:
                    new_token = ' '.join(token_list[0].split('$€£'))
                else:
                    new_token = token_list[0]
                data_temp_1.append({'string' : new_token, 'label' : token_list[1:]})
            else:
                data_temp_1.append({'string' : s_1_1[token], 'label' : []})
        data_1.append(data_temp_1)
     
    return data_1

if __name__ == '__main__':

    output_data = identify(data, sparv_list)
    