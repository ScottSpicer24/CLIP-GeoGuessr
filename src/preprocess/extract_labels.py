# stream the dataset and get the labels
import torch
from PIL import Image
from datasets import load_dataset
import os
import csv


def main():
    # Start streaming dataset
    # dataset = load_dataset('osv5m/osv5m', full=False, split='train', streaming=True, trust_remote_code=True) # Stream the data due to the size
    dataset = load_dataset('osv5m/osv5m', full=False, split='train', streaming=True) # Stream the data due to the size
    
    # Create a set and a csv path for each attribute
    country_set = set()
    region_set = set()
    sub_region_set = set()
    city_set = set()

    country_path = 'osv5m_countries.csv'
    region_path = 'osv5m_region.csv'
    sub_region_path = 'osv5m_sub_region.csv'
    city_path = 'osv5m_city.csv'

    # Go through each point in the dataset
    for i, data in enumerate(dataset):
        # Extract attribute value 
        country = data['country'].strip().lower()
        region = data['region'].strip().lower()
        sub_region = data['sub-region'].strip().lower()
        city = data['city'].strip().lower()

        # Each of the attribuites if NOT in set, add to set and print to csv
        if country not in country_set:
            country_set.add(country)
            csvPrint(country_path, country)
        
        if region not in region_set:
            region_set.add(region)
            csvPrint(region_path, region)

        if sub_region not in sub_region_set:
            sub_region_set.add(region)
            csvPrint(sub_region_path, sub_region)
        
        if city not in city_set:
            city_set.add(city)
            csvPrint(city_path, city)
        

# Print labels (country, continent, city...) to a csv
def csvPrint(path, item):
    # See if path exist, and append to it
    csv_exists = os.path.exists(path)
    attr = path.rstrip(".csv") # For the header
    with open(path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header if file doesn't exist
        if not csv_exists:
            writer.writerow([attr])
        
        # Write attribute values to csv
        writer.writerow([item])

# Run 
if __name__ == "__main__":
    main()   