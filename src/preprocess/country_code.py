import pycountry
import csv
import os

def main():
    path = './osv5m_countries.csv'
    
    # See if path exist
    csv_exists = os.path.exists(path)
    if not csv_exists:
        raise Exception("CSV path not found.") 

    # Read each of the country code
    rows = []
    with open(path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row[0]) != 2:
                continue
            # Append the full country name to the row
            # Try to get the country details
            country = pycountry.countries.get(alpha_2=row[0])
            if country:
                # Add the full country name
                # Use 'common_name' if available otherwise 'name'
                name = getattr(country, 'common_name', country.name)  
            else:
                name = "Unknown"  # Handle missing matches
            print(name)
            row.append(name)
            rows.append(row)
    
    # Write the new rows back to the same file
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


if __name__ == '__main__':
    main()
