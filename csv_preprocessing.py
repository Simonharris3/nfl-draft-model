import csv


def main():
    merge_data()
    # position_nums_to_letters()
    # test()


def test():
    file_r = open("Book1.csv", mode="r")
    reader = csv.reader(file_r)
    file_rep = []
    for row in reader:
        file_rep.append([])
        for num in row:
            file_rep[-1].append(num)
    file_r.close()

    for i in range(len(file_rep)):
        for c in range(len(file_rep[i])):
            if file_rep[i][c] == '6':
                print("found")
                file_rep[i][c] = '22'

    file_w = open("Book1.csv", mode="w")
    writer = csv.writer(file_w)
    for i in range(len(file_rep)):
        writer.writerow(file_rep[i])
    file_w.close()


def position_nums_to_letters():
    with open("sportsref_download_with_pff.csv", mode='r') as file:
        reader = csv.reader(file)
        file_rep = []
        for row in reader:
            if len(row) != 0:
                row_rep = row
                print(row)
                if is_num(row[1]):
                    # noinspection PyTypeChecker
                    row_rep[1] = nums_to_letters(row[1])

                for i in range(len(row[11:23])):
                    # make sure it's not the first row
                    if row[1] != "QB" and row[i + 11] != "" and row[1] != "Pos":
                        row_rep[i + 11] = ""

                file_rep.append(row_rep)

    with open("sportsref_download_with_pff.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for i in range(len(file_rep)):
            writer.writerow(file_rep[i])


def merge_data():
    base_file = open("sportsref_download_with_pff.csv", mode='r+')
    reader = csv.reader(base_file)
    next(reader)

    base_data = []
    for row in reader:
        base_data.append([])
        for datum in row:
            base_data[-1].append(datum)

    base_data = find_matches("defense_summary", base_data)
    base_data = find_matches("offense_blocking", base_data)
    base_data = find_matches("rushing_summary", base_data)
    base_data = find_matches("receiving_summary", base_data)

    # with open("sportsref_download_with_pff.csv", mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     for c in range(len(base_data)):
    #         writer.writerow(base_data[c])


# finds players in the pff data that also have combine data, and inserts the pff data into the combine file.
# file name is the start of the file name (ex. defense_summary)
# for each year from 2016 to 2021, the function will find the data with the start of that file name
# (ex. defense_summary_2016.csv)
# base_data is the combine data read in from sportsref_download_with_pff.csv
# when the program finds a match, it will insert the pff data into base_data in the proper place. it returns the
# updated version of base_data
def find_matches(file_name, base_data):
    files = []
    rb_snap_indices = (34, 35)
    input_snaps_index = -1
    ex_output = []

    # find where the snap count and grade is stored dependent on file
    if file_name == "defense_summary":
        input_grade_index = 12
        input_snaps_index = 31
    elif file_name == "offense_blocking":
        input_grade_index = 8
        input_snaps_index = 20
    elif file_name == "rushing_summary":
        input_grade_index = 23
    elif file_name == "receiving_summary":
        input_grade_index = 18
        input_snaps_index = 28
    else:
        raise ValueError("unknown file name: " + file_name)

    for i in range(6):
        files.append(open(file_name + "_20" + str(i + 16) + ".csv", mode='r'))
    # iterate through each file of the new data (one for every year)
    for year_index in range(6):
        reader = csv.reader(files[year_index])
        for row in reader:
            for base_index in range(len(base_data)):
                # if the name and position match
                if row[0] == base_data[base_index][0] and row[2] == base_data[base_index][1]:
                    # find where games played and the grade should be added for the current year
                    output_snaps_index = 11 + 2 * (5 - year_index)
                    output_grade_index = output_snaps_index + 1
                    if output_grade_index >= 23:
                        raise Exception(
                            "grade index should be less than 23, it is " + str(output_grade_index) + ". year "
                            "index is " + str(year_index))

                    # for rbs the snaps count isn't as clean, so we have to add 2 values
                    if input_snaps_index == -1:
                        snaps = row[rb_snap_indices[0]] + row[rb_snap_indices[1]]
                    else:
                        snaps = row[input_snaps_index]

                    # if there's already data there, that means the player has been counted as a different position
                    # in that case, we go with the position where the player had more snaps
                    if base_data[base_index][output_snaps_index] == "":
                        base_data[base_index][output_snaps_index] = snaps
                        base_data[base_index][output_grade_index] = row[input_grade_index]

                        if len(ex_output) < 3:
                            ex_output.append(base_data[base_index])
                    else:
                        if base_data[base_index][output_snaps_index] > snaps:
                            base_data[base_index][output_snaps_index] = snaps
                            base_data[base_index][output_grade_index] = row[input_grade_index]

    for i in range(6):
        files[i].close()

    for i in range(3):
        if i < len(ex_output):
            print(ex_output[i])

    print("\n")
    return base_data


def nums_to_letters(value):
    if value == '0':
        rvalue = 'QB'
    elif value == '1':
        rvalue = 'OT'
    elif value == '2':
        rvalue = 'OG'
    elif value == '3':
        rvalue = 'C'
    elif value == '4':
        rvalue = 'RB'
    elif value == '5':
        rvalue = 'TE'
    elif value == '6':
        rvalue = 'WR'
    elif value == '7':
        rvalue = 'DT'
    elif value == '8':
        rvalue = 'DL'
    elif value == '9':
        rvalue = 'EDGE'
    elif value == '10':
        rvalue = 'OLB'
    elif value == '11':
        rvalue = 'LB'
    elif value == '12':
        rvalue = 'CB'
    elif value == '13':
        rvalue = 'DB'
    elif value == '14':
        rvalue = 'S'
    elif value == '15':
        rvalue = 'P'
    elif value == '16':
        rvalue = 'K'
    elif value == '17':
        rvalue = 'LS'
    else:
        raise ValueError()

    return rvalue


# return whether a string is numeric
def is_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


main()
