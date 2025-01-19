import csv

def main():

    in_path = './src/preprocess/worldcities.csv'
    out_path = './src/preprocess/geolabels.csv'

    map = {}

    with open(in_path, mode='r', newline='', encoding='utf-8') as file:
        csvFile = csv.reader(file)
        i = 0
        for line in csvFile:
            #print(line[1], " : ", line[4])
            country = line[4]
            city = line[1]
            if country in map:
                map[country].append(city)
            else:
                map[country] = [city]


    with open(out_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        for key, value in map.items():
            entry = []
            for i, city in enumerate(value):
                '''if i == 30:
                    break'''
                entry.append(city)

            # Write attribute values to csv
            writer.writerow([key, entry])

if __name__ == "__main__":
    main()