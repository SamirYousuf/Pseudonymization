# Modules
import nltk
import re, random
import pandas as pd
import numpy as np
import argparse

# Pseudonymise the swedish cities and islands
# cities are replaced with other cities and islands are replaced with other islands
# Anonymise the postal code to "000 00"
def swe_cities_island_postal(data):
  list_swedish_cities = pd.read_csv('/data/cities_sweden.csv')
  list_cities = list_swedish_cities['Cities'].tolist()
  temp_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_cities:
        if j == k:
          y[i] = random.choice(list_cities)
    temp_list.append(' '.join(y))
  data = temp_list
  list_swedish_cities = pd.read_csv('/data/island_sweden.csv')
  list_cities = list_swedish_cities['Island'].tolist()
  temp_list = []
  for x in data:
    y = x.split(' ')
    for i, j in enumerate(y):
      for k in list_cities:
        if j == k:
          y[i] = random.choice(list_cities)
    temp_list.append(' '.join(y))
  data = temp_list
  temp_list = []
  for x in data:
    if re.search(r' \d{3} \d{2} ', x):
      x = re.sub(r'(\d{3} \d{2})', '000 00', x)
      temp_list.append(x)
    else:
      temp_list.append(x)
  return(' '.join(temp_list))


if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Program takes an input text file and output text file')
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--output', '-o', type=str, required=True)
  args = parser.parse_args()
  
  # Read a given file and save the tokenized sentence data in a list
  with open(args.input) as file:
    data = file.read()

  data = re.sub(r'\n\n', ' $$$$ . ', data)
  data = nltk.sent_tokenize(data)
  output_file = swe_cities_island_postal(data)
  output_file = output_file.replace(' $$$$ . ', '\n\n')

  if args.output:
    file = open(args.output, "w")
    file.write(output_file)
