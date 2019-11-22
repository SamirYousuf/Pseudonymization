# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
import argparse

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
def personel_data(data):
  result_list = []
  for x in data:
    if re.search(r' [A-Za-z]{3} \d{3} ', x):
      x = re.sub(r'(\w{3} \d{3})', 'ABC 000 ', x)
    if 'mobil' in x:
      y = x.split(' ')
      for i, j in enumerate(y):
        if re.search('\d{10}', j):
          y[i] = '0000-000000'
      result_list.append(' '.join(y))
    else:
      y = x.split(' ')
      for i, j in enumerate(y):
        if re.search(r'^\d{4}-\d{2}-\d{2}$', j):
          y[i] = '1111-11-11'
        if re.search(r'^\d{2}\/\d{2}\/\d{2}$', j):
          y[i] = '11/11/11'
        if re.search(r'^\d{6}$', j) or re.search(r'^\d{8}$', j):
          y[i] = '111111'
        if re.search(r'^\d{2}\.\d{2}\.\d{2}$', j):
          y[i] = '11.11.11'
        if re.search(r'^\d{1,2}/\d{2}$', j):
          y[i] = '11/11'
        if re.search(r'^\d{4}$', j):
          y[i] = str(int(j) + random.randint(-5,5))
        if re.search('\d{6}-\d{4}', j) or re.search('\d{8}-\d{4}', j) or re.search('\d{10}',j) or re.search('\d{12}', j):
          y[i] = '123456-0000'
        if '@' in j:
          y[i] = 'email@dot.com'
        if 'https' in j:
          y[i] = 'url.com'
        if 'http' in j:
          y[i] = 'url.com'
        if 'www' in j:
          y[i] = 'url.com'
      result_list.append(' '.join(y))
  new_data = []
  for x in result_list:
    if re.search(r' \d{4}-\d{2} \d{3} \d{2} ', x):
      x = re.sub(r'(\d{4}-\d{2} \d{3} \d{2} )', '0000-00 000 00 ', x)
      new_data.append(x)
    elif re.search(r' \d{4}-\d{3} \d{3} \d{3} ', x):
      x = re.sub(r'(\d{4}-\d{3} \d{3} \d{3} )', '0000-000 000 000 ', x)
      new_data.append(x)
    elif re.search(r' \d{4}-\d{1} \d{3} \d{3} \d{4} ', x):
      x = re.sub(r'(\d{4}-\d{1} \d{3} \d{3} \d{4} )', '0000-0 000 000 0000 ', x)
      new_data.append(x)
    elif re.search(r' \d{4} \d{2} \d{3} \d{2} ', x):
      x = re.sub(r'(\d{4} \d{2} \d{3} \d{2} )', '0000 00 000 00 ', x)
      new_data.append(x)
    elif re.search(r' \d{4} \d{3} \d{3} \d{3} ', x):
      x = re.sub(r'(\d{4} \d{3} \d{3} \d{3} )', '0000 000 000 000 ', x)
      new_data.append(x)
    elif re.search(r' \d{4} \d{1} \d{3} \d{3} \d{4} ', x):
      x = re.sub(r'(\d{4} \d{1} \d{3} \d{3} \d{4} )', '0000 0 000 000 0000 ', x)
      new_data.append(x)
    else:
      new_data.append(x)
  return(new_data)

# Changes months and days in the data
def days_months(data, list_days, list_months):
  result_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_days:
        if j == k:
          y[i] = random.choice(list_days)
    result_list.append(' '.join(y))
  data = result_list
  result_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_months:
        if j == k:
          y[i] = random.choice(list_months)
    result_list.append(' '.join(y))
  return(' '.join(result_list))

if __name__ == '__main__':
  
  list_days = ['måndag', 'tisdag', 'onsdag', 'torsdag', 'fredag', 'lordag', 'sondag']
  list_months = ['januari', 'februari', 'mars', 'april', 'maj', 'juni', 'juli', 'augusti', 'september', 'oktober', 'november', 'december']

  parser = argparse.ArgumentParser(description='Program takes an input text file and output text file')
  
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--output', '-o', type=str, required=True)
  args = parser.parse_args()
    
  # Read a given file and save the tokenized sentence data in a list
  with open(args.input) as file:
    data = file.read()

  data = re.sub(r'\n\n', ' $$$$ . ', data)
  data = nltk.sent_tokenize(data)
  data = personel_data(data)
  output_file = days_months(data, list_days, list_months)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, 'w')
    file.write(output_file)
