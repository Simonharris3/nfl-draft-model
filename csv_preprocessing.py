import csv

position_dict = {'QB': 0.0, 'OT': 1.0, 'OL': 2.0, 'OG': 2.0, 'C': 3.0, 'RB': 4.0, 'HB': 4.0, 'FB': 4.0, 'TE': 5.0,
                 'WR': 6.0, 'DT': 7.0, 'DL': 8.0, 'DE': 9.0, 'EDGE': 9.0, 'OLB': 10.0, 'LB': 11.0, 'CB': 12.0,
                 'DB': 13.0, 'S': 14.0, 'P': 15.0, 'K': 16.0, 'LS': 17.0}


def main():
    # merge_data()
    # position_nums_to_letters()
    # test()
    convert_to_percentile()


def convert_to_percentile():
    with open("sportsref_download_with_pff.csv", mode='r') as file:
        reader = csv.reader(file)
        next(reader)

        pos_vals = []
        for i in range(12):
            pos_vals.append([])
            for c in range(8):
                pos_vals[-1].append([])

        file_rep = []
        for row in reader:
            file_rep.append(row)
            for i in range(8):
                pos_num = percentile_pos_num(row[1])
                try:
                    datum = float(row[i+2])
                    if datum != 0:
                        pos_vals[pos_num][i].append(float(row[i + 2]))
                except ValueError:
                    if row[i+2] != '':
                        raise ValueError("wtf is this: " + row[i+2])

        for i in range(len(pos_vals)):
            for c in range(len(pos_vals[i])):
                pos_vals[i][c].sort()

        for row in file_rep:
            pos_num = percentile_pos_num(row[1])
            for i in range(len(row[2:10])):
                try:
                    ranked_list = pos_vals[pos_num][i]
                    datum = float(row[i+2])
                    rank_start = ranked_list.index(datum)
                    rank_end = rank_start
                    while ranked_list[rank_end] == rank_start:
                        rank_end += 1
                    rank = ((rank_end - 1) + rank_start) / 2
                    percentile = round(rank / len(ranked_list) * 100, 1)
                except ValueError:
                    percentile = -50
                row[i+2] = percentile

    with open("sportsref_download_with_pff.csv", mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(file_rep)):
            writer.writerow(file_rep[i])


def percentile_pos_num(pos):
    if pos == 'QB':
        return 0
    if pos == 'OT':
        return 1
    if pos == 'OL' or pos == 'OG' or pos == 'C':
        return 2
    if pos == 'RB' or pos == 'HB' or pos == 'FB':
        return 3
    if pos == 'TE':
        return 4
    if pos == 'WR':
        return 5
    if pos == 'DT' or pos == 'DL':
        return 6
    if pos == 'DE' or pos == 'EDGE' or pos == 'OLB':
        return 7
    if pos == 'LB':
        return 8
    if pos == 'CB':
        return 9
    if pos == 'DB' or pos == 'S':
        return 10
    if pos == 'P' or pos == 'K' or pos == 'LS':
        return 11


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
    base_data = find_matches("passing_summary", base_data)
    base_data = find_matches("rushing_summary", base_data)
    base_data = find_matches("receiving_summary", base_data)
    base_data = find_matches("offense_blocking", base_data)

    with open("sportsref_download_with_pff.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for c in range(len(base_data)):
            writer.writerow(base_data[c])


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
        input_snaps_index = 24
    elif file_name == "rushing_summary":
        input_grade_index = 23
    elif file_name == "receiving_summary":
        input_grade_index = 19
        input_snaps_index = 28
    elif file_name == "passing_summary":
        input_grade_index = 23
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

                if row[0] == base_data[base_index][0] and same_pos(row[2], base_data[base_index][1]):
                    # find where games played and the grade should be added for the current year
                    output_snaps_index = 11 + 2 * (5 - year_index)
                    output_grade_index = output_snaps_index + 1
                    if output_grade_index >= 23:
                        raise Exception(
                            "grade index should be less than 23, it is " + str(output_grade_index) + ". year "
                            "index is " + str(year_index))

                    # for rbs the snaps count isn't as clean, so we have to add 2 values
                    if input_snaps_index == -1:
                        snaps = float(row[rb_snap_indices[0]]) + float(row[rb_snap_indices[1]])
                    else:
                        snaps = float(row[input_snaps_index])

                    grade = row[input_grade_index]
                    player_data = base_data[base_index]
                    # if there's no data there, we insert the pff grade and snap count into the appropriate place
                    if player_data[output_snaps_index] == "":
                        base_data[base_index][output_snaps_index] = snaps
                        base_data[base_index][output_grade_index] = grade
                    else:
                        # if there's already data there, we have to determine which data to use. first, we determine
                        # whether a player is on offense or defense, and make sure the grade is derived from the
                        # appropriate file. if offense, the snap count should come from the blocking file if possible.
                        if is_offense(row[2]):
                            if file_name == "offense_blocking":
                                base_data[base_index][output_snaps_index] = snaps
                                base_data[base_index][output_grade_index] = grade
                            else:
                                if (row[2] == 'QB' and file_name == 'passing_summary') or \
                                        (row[2] == 'WR' and file_name == 'receiving_summary') or \
                                        (same_pos(row[2], 'RB') and file_name == "rushing_summary"):
                                    base_data[base_index][output_snaps_index] = snaps
                                    base_data[base_index][output_grade_index] = grade
                            # otherwise, we go with the data that better represents the players position.
                        else:
                            if file_name == "defense_summary":
                                base_data[base_index][output_snaps_index] = snaps
                                base_data[base_index][output_grade_index] = grade
                            # for defensive positions, we only get the data from the defense

                    # if len(ex_output) < 5:
                    #     ex_output.append(base_data[base_index][0:2] + base_data[base_index][11:])

    for i in range(6):
        files[i].close()

    print(file_name)
    for i in range(len(ex_output)):
        if i < len(ex_output):
            print(ex_output[i])

    print("\n")
    return base_data


# returns true if the 2 given positions are considered the same position
def same_pos(pos1, pos2):
    return pos1 == pos2 or \
        (pos1 == "OL" and (pos2 == 'OT' or pos2 == 'OG' or pos2 == 'C')) or \
        (pos2 == 'OL' and (pos1 == 'OT' or pos1 == 'OG' or pos1 == 'C')) or \
        (pos1 == 'RB' and (pos2 == 'HB' or pos2 == 'FB')) or \
        (pos2 == 'RB' and (pos1 == 'HB' or pos1 == 'FB')) or \
        (pos1 == 'DL' and (pos2 == 'DT' or pos2 == 'EDGE' or pos2 == 'DE' or pos2 == 'DL')) or \
        (pos2 == 'DL' and (pos1 == 'DT' or pos1 == 'EDGE' or pos1 == 'DE' or pos1 == 'DL')) or \
        (pos1 == 'EDGE' and (pos2 == 'OLB' or pos2 == 'DE')) or (pos2 == 'EDGE' and (pos1 == 'OLB' or pos1 == 'DE')) \
        or (pos1 == 'OLB' and pos2 == 'LB') or (pos2 == 'LB' and pos1 == 'OLB') or \
        (pos1 == 'DB' and (pos2 == 'S' or pos2 == 'CB')) or (pos2 == 'DB' and (pos1 == 'S' or pos1 == 'CB'))


# returns true if the given position is an offensive position
def is_offense(pos):
    return pos == 'QB' or pos == 'OL' or pos == 'OT' or pos == 'OG' or pos == 'C' \
        or pos == 'RB' or pos == 'HB' or pos == 'FB' or pos == 'TE' or pos == 'WR'


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
