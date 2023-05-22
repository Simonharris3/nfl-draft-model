import csv

position_dict = {'QB': 0.0, 'OT': 1.0, 'T': 1.0, 'OL': 2.0, 'OG': 2.0, 'G': 2.0, 'C': 3.0, 'RB': 4.0, 'HB': 4.0,
                 'FB': 4.0, 'TE': 5.0, 'WR': 6.0, 'DT': 7.0, 'DI': 7.0, 'DL': 8.0, 'DE': 9.0, 'EDGE': 9.0, 'ED': 9.0,
                 'OLB': 10.0, 'LB': 11.0, 'CB': 12.0, 'DB': 13.0, 'S': 14.0, 'P': 15.0, 'K': 16.0, 'LS': 17.0, 'ST': 18,
                 '': 19.0}

pff_data_indices = {"defense_summary": (12, 31), "passing_summary": (23, 28), "receiving_summary": (19, 28),
                    "offense_blocking": (8, 24)}
rushing_indices = (23, (34, 35))

label_row = ["Name", "Pos", "Height", "Weight", "40 time", "Vert", "Bench",	"Broad", "3cone", "Shuttle", "2021 snaps",
             "2021 grade", "2020 snaps", "2020 grade", "2019 snaps", "2019 grade", "2018 snaps", "2018 grade",
             "2017 snaps", "2017 grade", "2016 snaps", "2016 grade", "Pick", "School"]

pos_groups = [['QB'],
              ['OT', 'T', 'OL', 'OG', 'G', 'C'],
              ['RB', 'HB', 'FB'],
              ['TE'],
              ['WR'],
              ['DT', 'DI', 'DL'],
              ['DE', 'DL', 'ED', 'EDGE', 'OLB', 'LB'],
              ['CB', 'DB', 'S'],
              ['P'],
              ['K', 'PK'],
              ['LS'],
              ['ST']]

name_pairs = [['josh', 'joshua'],
              ['jeff', 'jeffrey'],
              ['chris', 'christopher']]

school_pairs = [['alab a&m', 'alabama a&m'],
                ['app state', 'appalachian state'],
                ['ark state', 'arkansas state'],
                ['boston col.', 'boston col'],
                ['bowling green', 'bowl green'],
                ['central michigan', 'c michigan'],
                ['central florida', 'ucf'],
                ['cal', 'california'],
                ['cent ark', 'central arkansas'],
                ['charleston southern', 'charles so'],
                ['colo state', 'colorado state'],
                ['east carolina', 'e carolina'],
                ['east. washington', 'e washgton'],
                ['florida atlantic', 'fau'],
                ['ga state', 'georgia state'],
                ['georgia tech', 'ga tech'],
                ['illinois state', 'ill state'],
                ['jacksonville state', 'jville state'],
                ['louisiana-lafayette', 'la lafayet'],
                ['louisiana tech', 'la tech'],
                ['miami (fl)', 'miami fl'],
                ['miami (oh)', 'miami oh'],
                ['mich state', 'michigan state'],
                ['miss state', 'mississippi state'],
                ['missouri state', 'mo state'],
                ['n carolina', 'north carolina'],
                ['n dak st', 'north dakota st'],
                ['n illinois', 'northern illinois'],
                ['n texas', 'north texas'],
                ['nc state', 'north carolina state'],
                ['new mex state', 'new mexico state'],
                ['nwestern', 'northwestern'],
                ['okla state', 'oklahoma state'],
                ['mississippi', 'ole miss'],
                ['south alabama', 's alabama'],
                ['s carolina', 'south carolina'],
                ['south dakota state', 's dak st'],
                ['usf', 'south florida'],
                ['s jose st', 'san jose state'],
                ['s diego st', 'san diego state'],
                ['so miss', 'southern miss'],
                ['stf austin', 'stephen f. austin'],
                ['stny brook', 'stony brook'],
                ['tenn state', 'tennessee state'],
                ['texas-san antonio', 'utsa'],
                ['utep', 'texas-el paso'],
                ['virginia tech', 'va tech'],
                ['western kentucky', 'w kentucky'],
                ['w georgia', 'western georgia'],
                ['w virginia', 'west virginia'],
                ['w michigan', 'western michigan'],
                ['wake forest', 'wake'],
                ['wash state', 'washington state'],
                ['wm & mary', 'william & mary'],
                ['youngstown state', 'yngtown st']]


def main():
    # fix_empty_pick_num()
    # get_rid_of_whitespace()

    # with open("combine data/all_combine_data.csv", mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(label_row)
    #
    # for i in range(7):
    #     add_combine_data(str(i+2017))
    #
    remove_special_teams()

    # convert_to_percentile()
    # merge_production_data()


def add_combine_data(year):
    data = []
    with open("combine data/combine_data_" + year + ".csv", mode='r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            row_rep = row
            if row_rep[-1] == '':
                row_rep[-1] = '300'

            if is_num(row_rep[3]):
                ht = float(row_rep[3])
                if ht > 1000:
                    row_rep[3] = date_to_height(ht)

            zeros = []
            # leave space for grade data
            for i in range(12):
                zeros.append('0')

            row_rep = row_rep[0:2] + row_rep[3:-1] + zeros + [row_rep[-1], row_rep[2]]
            data.append(row_rep)

    with open("combine data/all_combine_data.csv", mode='a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


def date_to_height(num):
    if num == 36678:
        return '72'
    elif num < 45078:
        return str(num - 44986)
    else:
        return str(num - 45005)


# remove all solely special teams players
def remove_special_teams():
    with open("sportsref_with_pff_new.csv", mode='r') as file:
        file_rep = []
        reader = csv.reader(file)

        for row in reader:
            if row[1] != 'K' and row[1] != 'P' and row[1] != 'PK' and row[1] != 'LS' and row[1] != 'ST':
                file_rep.append(row)

    with open("sportsref_with_pff_new.csv", mode='w', newline='') as file:
        write_file(file_rep, file)


def fix_empty_pick_num():
    file_rep = []
    with open("sportsref_with_pff_new.csv", mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            if row[-2] == '':
                row[-2] = '300'
            file_rep.append(row)

        write_file(file_rep, file)


def write_file(file_rep, file):
    writer = csv.writer(file)
    for i in range(len(file_rep)):
        writer.writerow(file_rep[i])


def get_rid_of_whitespace():
    file_rep = []
    with open("combine data/all_combine_data.csv", mode='r') as file:
        reader = csv.reader(file)

        for row in reader:
            if len(row) > 0:
                file_rep.append(row)

    with open("combine data/all_combine_data.csv", mode='w', newline='') as file:
        write_file(file_rep, file)


def convert_to_percentile():
    with open("combine data/all_combine_data.csv", mode='r') as file:
        reader = csv.reader(file)

        pos_vals = []
        for i in range(12):
            pos_vals.append([])
            for c in range(8):
                pos_vals[-1].append([])

        file_rep = []
        for row in reader:
            file_rep.append(row)
            if row[0] != "Name":
                for i in range(8):
                    pos_num = percentile_pos_num(row[1])
                    try:
                        datum = float(row[i + 2])
                        if datum != 0:
                            pos_vals[pos_num][i].append(float(row[i + 2]))
                    except ValueError:
                        if row[i + 2] != '':
                            raise ValueError("wtf is this: " + row[i + 2])

        for i in range(len(pos_vals)):
            for c in range(len(pos_vals[i])):
                pos_vals[i][c].sort()

        for row in file_rep:
            if row[0] != "Name":
                pos_num = percentile_pos_num(row[1])
                for i in range(len(row[2:10])):
                    try:
                        ranked_list = pos_vals[pos_num][i]
                        datum = float(row[i + 2])
                        rank_start = ranked_list.index(datum)
                        rank_end = rank_start
                        while ranked_list[rank_end] == ranked_list[rank_start] and rank_end+1 < len(ranked_list):
                            rank_end += 1
                        if rank_end > rank_start:
                            rank = ((rank_end - 1) + rank_start) / 2
                        else:
                            rank = rank_start

                        percentile = round(rank / len(ranked_list) * 100, 1)

                        if percentile < 0:
                            raise Exception("negative percentile")
                    except ValueError:
                        percentile = -50
                    row[i + 2] = percentile

    base_file = open("sportsref_with_pff_new.csv", mode='w', newline='')
    write_file(file_rep, base_file)


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
    if pos == 'DE' or pos == 'EDGE' or pos == 'OLB' or pos == 'ED':
        return 7
    if pos == 'LB' or pos == 'ILB':
        return 8
    if pos == 'CB':
        return 9
    if pos == 'DB' or pos == 'S':
        return 10
    if pos == 'P' or pos == 'K' or pos == 'LS':
        return 11
    else:
        raise ValueError("unexpected position: " + pos)


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

    file_w = open("Book1.csv", mode="w", newline='')
    writer = csv.writer(file_w)
    for i in range(len(file_rep)):
        writer.writerow(file_rep[i])
    file_w.close()


def merge_production_data():
    base_file = open("sportsref_with_pff_new.csv", mode='r+')
    reader = csv.reader(base_file)

    base_data = []
    for row in reader:
        base_data.append(row)

    base_data = find_matches("defense_summary", base_data)
    base_data = find_matches("passing_summary", base_data)
    base_data = find_matches("rushing_summary", base_data)
    base_data = find_matches("receiving_summary", base_data)
    base_data = find_matches("offense_blocking", base_data)

    with open("sportsref_with_pff_new.csv", mode='w', newline='') as file:
        write_file(base_data, file)


# finds players in the pff data that also have combine data, and inserts the pff data into the combine file.
# file name is the start of the file name (ex. defense_summary)
# for each year from 2016 to 2021, the function will find the data with the start of that file name
# (ex. defense_summary_2016.csv)
# base_data is the combine data read in from sportsref_with_pff_new.csv
# when the program finds a match, it will insert the pff data into base_data in the proper place. it returns the
# updated version of base_data
def find_matches(file_name, base_data):
    files = []
    ex_output = []
    input_snaps_index = -1
    rb_snap_indices = None

    # find where the snap count and grade is stored dependent on file
    try:
        input_grade_index, input_snaps_index = pff_data_indices[file_name]
    except KeyError:
        if file_name == "rushing_summary":
            rushing = True
            input_grade_index = rushing_indices[0]
            rb_snap_indices = rushing_indices[1]
        else:
            raise ValueError("unknown file name: " + file_name)

    for i in range(6):
        files.append(open("pff data/" + file_name + "_20" + str(i + 16) + ".csv", mode='r'))

    # iterate through each file of the new data (one for every year)
    for year_index in range(6):
        reader = csv.reader(files[year_index])
        for row in reader:
            for base_index in range(len(base_data)):
                try:
                    k = position_dict[row[2]]
                except KeyError:
                    if row[2] != 'position':
                        raise ValueError("unknown position: " + row[2] + "; player: " + row[0] +
                                         "; file name: " + file_name)
                # if the name and school match
                if same_name(row[0], base_data[base_index][0]) and same_school(row[3], base_data[base_index][-1]):
                    # find where games played and the grade should be added for the current year
                    output_snaps_index = 10 + 2 * (5 - year_index)
                    output_grade_index = output_snaps_index + 1
                    if output_grade_index >= 23:
                        raise Exception(
                            "grade index should be less than 23, it is " + str(output_grade_index) + ". year "
                            "index is " + str(year_index))

                    # for the rushing file the snap count isn't as clean, so we have to add 2 values
                    if rb_snap_indices:
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
                        else:
                            # for defensive positions, we only get the data from the defense
                            if file_name == "defense_summary":
                                base_data[base_index][output_snaps_index] = snaps
                                base_data[base_index][output_grade_index] = grade

                    # if len(ex_output) < 5:
                    #     ex_output.append(base_data[base_index][0:2] + base_data[base_index][11:])

    for i in range(6):
        files[i].close()

    for i in range(len(ex_output)):
        if i < len(ex_output):
            print(ex_output[i])

    return base_data


def position_nums_to_letters():
    with open("sportsref_with_pff_new.csv", mode='r') as file:
        reader = csv.reader(file)
        file_rep = []
        for row in reader:
            if len(row) != 0:
                row_rep = row
                if is_num(row[1]):
                    # noinspection PyTypeChecker
                    row_rep[1] = nums_to_letters(row[1])

                for i in range(len(row[11:23])):
                    # make sure it's not the first row
                    if row[1] != "QB" and row[i + 11] != "" and row[1] != "Pos":
                        row_rep[i + 11] = ""

                file_rep.append(row_rep)

    with open("sportsref_with_pff_new.csv", mode='w', newline='') as file:
        write_file(file_rep, file)


# returns true if the 2 given positions are considered the same position
def same_pos(pos1, pos2):
    for pos_group in pos_groups:
        if pos1 in pos_group and pos2 in pos_group:
            return True
    return pos1 == '' or pos2 == '' or pos1 == 'ST' and (pos2 == 'P' or pos2 == 'K' or pos2 == 'ST')


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


def same_name(name1, name2):
    n1 = name1.lower()
    n2 = name2.lower()
    names1 = n1.split(" ")
    names2 = n2.split(" ")
    for i in range(len(name_pairs)):
        if names1[0] in name_pairs[i] and names2[0] in name_pairs[i]:
            if same_name(names1[1], names2[1]):
                return True

    return n1 == n2 or n1 == n2 + " jr." or n2 == n1 + " jr." or \
        n1 == n2 + " ii" or n2 == n1 + " ii" or n1 == n2 + " iii" or n2 == n1 + " iii"


def same_school(school1, school2):
    s1 = school1.lower()
    s2 = school2.lower()

    for i in range(len(school_pairs)):
        if s1 in school_pairs[i] and s2 in school_pairs[i]:
            return True

    words1 = school1.split(" ")
    words2 = school2.split(" ")

    if words1[:-1] == words2[:-1] and ((words1[-1] == "state" and words2[-1] == "st") or
                                       (words1[-1] == "st" and words2[-1] == "state")):
        return True

    return s1 == s2


main()
