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

with open(MODULEDIR + '/dataset/names_database.json', 'r') as file:
        dict_names = json.load(file)

        
list_job_title = pd.read_csv(MODULEDIR + '/dataset/Prof_dataset.csv')
list_data = pd.read_csv(MODULEDIR + '/dataset/city_country.csv')
list_swedish_cities = pd.read_csv(MODULEDIR + '/dataset/cities_sweden.csv')
list_swedish_island = pd.read_csv(MODULEDIR + '/dataset/island_sweden.csv')

# Main function to de-identify all the personal information and save as a text file
def identify(data):
    # To have the output format same as the input especially for newline and paragraphs in the text
    data = re.sub(r'\n\n', ' $$$$ . ', data)  
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
   
    _data = []
    
    # Swedish bank account format
    for line in data:
        if re.search(r' \d{4}-\d{2} \d{3} \d{2} ', line):
            line = re.sub(r'(\d{4}-\d{2} \d{3} \d{2} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000-00 000 00
            _data.append(line)
        elif re.search(r' \d{4}-\d{3} \d{3} \d{3} ', line):
            line = re.sub(r'(\d{4}-\d{3} \d{3} \d{3} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000-000 000 000
            _data.append(line)
        elif re.search(r' \d{4}-\d{1} \d{3} \d{3} \d{4} ', line):
            line = re.sub(r'(\d{4}-\d{1} \d{3} \d{3} \d{4} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000-0 000 000 0000
            _data.append(line)
        elif re.search(r' \d{4} \d{2} \d{3} \d{2} ', line):
            line = re.sub(r'(\d{4} \d{2} \d{3} \d{2} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000 00 000 00
            _data.append(line)
        elif re.search(r' \d{4} \d{3} \d{3} \d{3} ', line):
            line = re.sub(r'(\d{4} \d{3} \d{3} \d{3} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000 000 000 000
            _data.append(line)
        elif re.search(r' \d{4} \d{1} \d{3} \d{3} \d{4} ', line):
            line = re.sub(r'(\d{4} \d{1} \d{3} \d{3} \d{4} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000 0 000 000 0000
            _data.append(line)
        elif re.search(r' \d{11} ', line):
            line = re.sub(r'(\d{11} )', '<bank_acc>bank_acc</bank_acc> ', line) # 00000000000
            _data.append(line)
        elif re.search(r' \d{13} \d{1} \d{3} \d{3} \d{4} ', line):
            line = re.sub(r'(\d{13} )', '<bank_acc>bank_acc</bank_acc> ', line) # 0000000000000
            _data.append(line)
        elif re.search(r' \d{15} \d{1} \d{3} \d{3} \d{4} ', line):
            line = re.sub(r'(\d{15} )', '<bank_acc>bank_acc</bank_acc> ', line) # 000000000000000
            _data.append(line)
        else:
            _data.append(line)
    
    data = _data
    _data = []
    
    for line in data:
        if re.search(r' [A-Za-z]{3} \d{3} ', line):  # Vehicle License number
            line = re.sub(r'(\w{3} \d{3})', '<license_nr>license_nr</license_nr> ', line) # ABC 000
        if re.search(r' \d{3}-\d{6} ', line):  # Landline number in Sweden
            line = re.sub(r'(\d{3}-\d{6})', '<landline_nr>landline_nr</landline_nr> ', line) # 000-000000
        if 'mobil' in line:  # Mobile number format
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if re.search('\d{10}', j):
                    line_split[i] = '<mobile_nr>mobile_nr</mobile_nr>' # 0000-000000
            _data.append(' '.join(line_split))
        else:
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if re.search(r'^\d{4}-\d{2}-\d{2}$', j):  # Date 1111-11-11
                    line_split[i] = '<date_type>date_type</date_type>'
                if re.search(r'^\d{2}\/\d{2}\/\d{2}$', j):  # Date 11/11/11
                    line_split[i] = '<date_type>date_type</date_type>'
                if re.search(r'^\d{6}$', j) or re.search(r'^\d{8}$', j):  # Date 111111
                    line_split[i] = '<date_type>date_type</date_type>'
                if re.search(r'^\d{2}\.\d{2}\.\d{2}$', j):  # Date 11.11.11
                    line_split[i] = '<date_type>date_type</date_type>'
                if re.search(r'^\d{1,2}/\d{2}$', j):  # Date 11/11
                    line_split[i] = '<date_type>date_type<date_type>'
                if re.search(r'^\d{4}$', j):  # Year - randomise "2018" with (-2,2) # If statement
                    line_split[i] = '<year_type>year_type</year_type>'
                # Personal number formats
                if re.search('\d{6}-\d{4}', j) or re.search('\d{8}-\d{4}', j) or re.search('\d{10}',j) or re.search('\d{12}', j):
                    line_split[i] = '<personal_id>personal_id</personal_id>' # 123456-0000
                if '@' in j:  # Email addresses are formatted to email @dot.com
                    line_split[i] = '<email_address>email_address</email_address>' # email@dot.com
                if 'https' in j:  # https url format
                    line_split[i] = '<url_link>url_link</url_link>' # url.com
                if 'http' in j: # http url format
                    line_split[i] = '<url_link>url_link</url_link>' # url.com
                if 'www' in j:  # www web address
                    line_split[i] = '<url_link>url_link</url_link>' # url.com
            _data.append(' '.join(line_split))
            
    data = _data
    _data = []
           
    # Randomised days in the data using a list of all the days in a week
    for line in data:
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_days:
                if j == k:
                    line_split[i] = '<day_type>day_type</day_type>' # random.choice(list_days)
        _data.append(' '.join(line_split))
    data = _data
    _data = []
    
    # Randomised months loop
    for line in data:
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_months:
                if j == k:
                    line_split[i] = '<month_type>month_type</montt_type>' # random.choice(list_months)
        _data.append(' '.join(line_split))
     
    data = _data
    _data = []
    
    for line in data:
        if re.search(r'\är \d{2} ', line): # Age mentioned in numbers are randomised with (-2,2)
            y = re.findall(r'\är (\d{2}) ', line)
            y1 = str(int(y[0]) + random.randint(-2,2))
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0]:
                    line_split[i] = '<age_digit>age_digit</age_digit>'
            _data.append(' '.join(line_split))
        elif len(re.findall(r'(fylla|fyller|fyllde|fyllt) (\d{2}) ', line)) > 0: 
            # Age mentioned in numbers are randomised with (-2,2) using RegEx
            y = re.findall(r'\d{2}', line)
            y1 = str(int(y[0]) + random.randint(-2,2))
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0]:
                    line_split[i] = '<age_digit>age_digit</age_digit>'
            _data.append(' '.join(line_split))
        elif len(re.findall(r'(fylla|fyller|fyllde|fyllt) (\w+) ', line)) > 0:
            # Age mentioned in words are randomised with (-2,2)
            # If the age is misspelled then it is autocorrected and then randomised
            list_number = tuple([key for key, value in dict_numbers.items()])
            y = re.findall(r'(fylla|fyller|fyllde|fyllt) (\w+) ', line)
            y1 = list(y[0])
            y2 = get_correct_spelling(y1[1], list_number)
            y3 = dict_numbers[y2]
            y3 = str(int(y3) + random.randint(-2,2))
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y1[1]:
                    line_split[i] = '<age_string>age_string</age_string>'
            _data.append(' '.join(line_split))
        else:
            _data.append(line)
            
    data = _data
    _data = []
    
    for line in data:
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_family:
                if j == k:
                    line_split[i] = '<family_info>family_info</family_info>' # random.choice(list_family)
        _data.append(' '.join(line_split))
    
    data = _data
    _data = []
    
    for line in data:
        if re.search(r'kompisar', line):
            y = re.findall(r' ([\w]+) kompisar', line)
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0]:
                    line_split[i] = '<family_info>family_info</family_info>' # random.choice(list_siblings)
            _data.append(' '.join(line_split))
        elif re.search(r'bröder', line):
            y = re.findall(r' ([\w]+) bröder', line)
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0]:
                    line_split[i] = '<family_info>family_info</family_info>' # random.choice(list_siblings)
            _data.append(' '.join(line_split))
        elif re.search(r'systern', line):
            y = re.findall(r' ([\w]+) systern', line)
            line_split = line.split(' ')
            for i, j in enumerate(line_split):
                if j == y[0]:
                    line_split[i] = '<family_info>family_info</family_info>' # random.choice(list_siblings)
            _data.append(' '.join(line_split))
        else:
            _data.append(line)
            
    data = _data
    _data = []
    
    for line in data:
        for y,z in dict_universities.items():
            for i in z:
                if i in line:
                    line = line.replace(i, '<university_name>'+y+'</university_name>')
        _data.append(line)
        
    data = _data
    _list = {}
    _data = []
    
    list_title = list_job_title['Yrkesbenämning'].tolist()
    list_title = list_title[:-7]
    
    for x in range(len(list_title)):
        if ',' in list_title[x]:
            y = list_title[x].split(',')
            _list[x] = y[0].lower()
        else:
            _list[x] = list_title[x].lower()
            
    for line in data:
        for key,value in _list.items():
            if value in line.split(' '):
                line = line.replace(value, '<prof_id>prof_id'+str(key)+'</prof_id>')
        _data.append(line)

        
    data = _data
    country_in_data = {}
    cities_in_data = {}
    
    list_country = list_data['Countries'].tolist()
    list_city = list_data['Cities'].tolist()
    
    for i in data:
        for j in set(list_country):
            if str(j) in i:
                country_in_data[list_country.index(str(j))] = j
        for k in set(list_city):
            if k in i.split(' '):
                cities_in_data[list_city.index(k)] = k
                
    _data = ' '.join(data)
    
    if len(country_in_data) > 0:
        if len(cities_in_data) > 0:
            for i,j in cities_in_data.items():
                index = list_city.index(j)
                if list_country[index] in country_in_data.values():
                    x = random.choice(list_country)
                    _data = _data.replace(list_country[index], '<country_name>country_name-'+str(index)+'</country_name>')
                    _index = list_country.index(x)
                    _data = _data.replace(list_city[index], '<city_name>city_name-'+str(index)+'</city_name>')
                else:
                    _data = _data.replace(j, '<city_name>city_name-'+str(i)+'</city_name>')
            for i,j in country_in_data.items():
                index = list_country.index(j)
                if list_city[index] in cities_in_data.values():
                    x = random.choice(list_city)
                    _data = _data.replace(list_city[index], '<city_name>city_name-'+str(index)+'</city_name>')
                    _index = list_city.index(x)
                    _data = _data.replace(list_country[index], '<country_name>country_name-'+str(index)+'</country_name>')
                else:
                    _data = _data.replace(j, '<country_name>country_name-'+str(i)+'</country_name>')
    elif len(cities_in_data) > 0:
        for i,j in cities_in_data.items():
            _data = _data.replace(j, '<city_name>city_name-'+str(i)+'</city_name>')
            
    data = nltk.sent_tokenize(_data)  
    _data = []

    list_cities = list_swedish_cities['Cities'].tolist()
    
    for line in data:  # Cities
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_cities:
                if j == k:
                    line_split[i] = '<swedish_city>swedish_city-'+str(list_cities.index(k))+'</swedish_city>' 
                    # random.choice(list_cities)
        _data.append(' '.join(line_split))
        
    data = _data
    _data = []
    list_cities = list_swedish_island['Island'].tolist()
    
    for line in data:  # Island
        line_split = line.split(' ')
        for i, j in enumerate(line_split):
            for k in list_cities:
                if j == k:
                    line_split[i] = '<swedish_island>swedish_island-'+str(list_cities.index(k))+'</swedish_island>' 
                    # random.choice(list_cities)
        _data.append(' '.join(line_split))
        
    data = _data
    _data = []
    
    for line in data:  # Postal Code only swedish
        if re.search(r' \d{3} \d{2} ', line):
            line = re.sub(r'(\d{3} \d{2})', '<postal_code>postal_code</postal_code>', line) # 000 00
            _data.append(line)
        else:
            _data.append(line)

    data = _data
   
    dict_tilltal_man = {}
    dict_tilltal_kvn = {}
    dict_fornamn_man = {}
    dict_fornamn_kvn = {}
    dict_efternamn = {}

    for line in data:
        for i in range(len(dict_names['tilltal_män'][0])):
            if dict_names['tilltal_män'][0][i] in line.split(' '):
                if i not in dict_tilltal_man:
                    dict_tilltal_man[i] = (len(dict_tilltal_man.keys())+1, dict_names['tilltal_män'][0][i])
                else:
                    pass
        for i in range(len(dict_names['förnamn_män'][0])):
            if dict_names['förnamn_män'][0][i] in line.split(' '):
                if i not in dict_fornamn_man:
                    dict_fornamn_man[i] = (len(dict_fornamn_man.keys())+1, dict_names['förnamn_män'][0][i])
                else:
                    pass
        for i in range(len(dict_names['efternamn'][0])):
            if dict_names['efternamn'][0][i] in line.split(' '):
                if i not in dict_efternamn:
                    dict_efternamn[i] = (len(dict_efternamn.keys())+1, dict_names['efternamn'][0][i])
                else:
                    pass
        for i in range(len(dict_names['tilltal_kvinnor'][0])):
            if dict_names['tilltal_kvinnor'][0][i] in line.split(' '):
                if i not in dict_tilltal_kvn:
                    dict_tilltal_kvn[i] = (len(dict_tilltal_kvn.keys())+1, dict_names['tilltal_kvinnor'][0][i])
                else:
                    pass
        for i in range(len(dict_names['förnamn_kvinnor'][0])):
            if dict_names['förnamn_kvinnor'][0][i] in line.split(' '):
                if i not in dict_fornamn_kvn:
                    dict_fornamn_kvn[i] = (len(dict_fornamn_kvn.keys())+1, dict_names['förnamn_kvinnor'][0][i])
                else:
                    pass

    _data = ' '.join(data)
    data = _data.split(' ')

    for i,j in enumerate(data):
        for key,value in dict_fornamn_man.items():
            if value[1] == j:
                if data[i-1] in dict_names['förnamn_män'][0]:
                    data[i-1] = '<fornamn_man>index-'+str(dict_names['förnamn_män'][0].index(data[i-1]))+'-'+str(value[0])+'</fornamn_man>'
                    data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'</fornamn_man>'
                    if data[i+1] in dict_names['efternamn'][0]:
                        data[i+1] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'
                    if data[i+1] in dict_names['förnamn_man'][0]:
                        data[i+1] = '<fornamn_man>index-'+str(dict_names['förnamn_män'][0].index(data[i+1]))+'-'+str(value[0])+'</fornamn_man>'
                        if data[i+2] in dict_names['efternamn'][0]:
                            data[i+2] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'                
                elif data[i-1] not in dict_names['förnamn_män'][0]:
                    if data[i+1] in dict_names['förnamn_män'][0]:
                        data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'</fornamn_man>'
                        data[i+1] = '<fornamn_man>index-'+str(dict_names['förnamn_män'][0].index(data[i+1]))+'-'+str(value[0])+'</fornamn_man>'
                        if data[i+2] in dict_names['efternamn'][0]:
                            data[i+2] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+2]))+'-'+str(value[0])+'</efternamn>'
                    elif data[i+1] in dict_names['efternamn'][0]:
                        data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'</fornamn_man>'
                        data[i+1] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'
                    else:
                        data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'</fornamn_man>'

    for i,j in enumerate(data):
        for key,value in dict_fornamn_kvn.items():
            if value[1] == j:
                if data[i-1] in dict_names['förnamn_kvinnor'][0]:
                    data[i-1] = '<fornamn_kvinnor>index-'+str(dict_names['förnamn_kvinnor'][0].index(data[i-1]))+'-'+str(value[0])+'</fornamn_kvinnor>'
                    data[i] = '<fornamn_kvinnor>index-'+str(key)+'-'+str(value[0])+'</fornamn_kvinnor>'
                    if data[i+1] in dict_names['efternamn'][0]:
                        data[i+1] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'
                    if data[i+1] in dict_names['fornamn_kvinnor'][0]:
                        data[i+1] = '<fornamn_kvinnor>index-'+str(dict_names['fornamn_kvinnor'][0].index(data[i+1]))+'-'+str(value[0])+'</fornamn_kvinnor>'
                        if data[i+2] in dict_names['efternamn'][0]:
                            data[i+2] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'                
                elif data[i-1] not in dict_names['förnamn_kvinnor'][0]:
                    if data[i+1] in dict_names['förnamn_kvinnor'][0]:
                        data[i] = '<fornamn_kvinnor>index-'+str(dict_names['förnamn_kvinnor'][0].index(data[i]))+'-'+str(value[0])+'</fornamn_kvinnor>'
                        data[i+1] = '<fornamn_kvinnor>index-'+str(dict_names['förnamn_kvinnor'][0].index(data[i+1]))+'-'+str(value[0])+'</fornamn_kvinnor>'
                        if data[i+2] in dict_names['efternamn'][0]:
                            data[i+2] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+2]))+'-'+str(value[0])+'</efternamn>'
                    elif data[i+1] in dict_names['efternamn'][0]:
                        data[i] = '<fornamn_kvinnor>index-'+str(key)+'-'+str(value[0])+'</fornamn_kvinnor>'
                        data[i+1] = '<efternamn>index-'+str(dict_names['efternamn'][0].index(data[i+1]))+'-'+str(value[0])+'</efternamn>'
                    else:
                        data[i] = '<fornamn_kvinnor>index-'+str(key)+'-'+str(value[0])+'</fornamn_kvinnor>'


    for i,j in enumerate(data):                   
        for key, value in dict_fornamn_man.items():
            if value[1] == j:
                data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'</fornamn_man>'

    for i,j in enumerate(data):                   
        for key, value in dict_fornamn_kvn.items():
            if value[1] == j:
                data[i] = '<fornamn_kvinnor>index-'+str(key)+'-'+str(value[0])+'</fornamn_kvinnor>'

    for i,j in enumerate(data):
        for key, value in dict_efternamn.items():
            if value[1] == j:
                data[i] = '<efternamn>index-'+str(key)+'-'+str(value[0])+'</efternamn>'

    for i,j in enumerate(data):
        for key, value in dict_fornamn_man.items():
            value_s = value[1] + 's'
            if value_s == j:
                data[i] = '<fornamn_man>index-'+str(key)+'-'+str(value[0])+'-gen</fornamn_man>'

    for i,j in enumerate(data):
        for key, value in dict_fornamn_kvn.items():
            value_s = value[1] + 's'
            if value_s == j:
                data[i] = '<fornamn_kvinnor>index-'+str(key)+'-'+str(value[0])+'-gen</fornamn_kvinnor>'

    data = ' '.join(data)            

    output_data = data.replace(' $$$$ . ', '\n\n')
    
    return output_data
    
    
if __name__ == '__main__':

    output_data = identify(data)
