#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
#sys.path.insert(1, '/Users/John/Documents/GitHub/ud120-projects/
enron_data = pickle.load(open("/Users/John/Documents/GitHub/ud120-projects/final_project/final_project_dataset.pkl", "rb"))

print(len(enron_data))
