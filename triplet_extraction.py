import random
import csv
import glob
import sys

folder = sys.argv[1]

list_aa = ["A", "R", "N", "D", "C", "Q", "E", "G", "H",
           "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
dict_aa = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9,
           "L": 10, "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19}
dict_all = {}

list_aa_combined = [[""]*20 for i in range(20)]
list_all = [""]*801
id_list = 0
list_all[id_list] = "label"
csv_file = open("%s_data.csv" % folder, "w", newline='')
csvwriter = csv.writer(csv_file)

id_col = 1
for i in range(20):
    for j in range(20):
        list_aa_combined[i][j] = list_aa[i] + list_aa[j]
        id_list += 1
        list_all[id_list] = list_aa_combined[i][j]
        id_col += 1
id_list=1
for i in range(800):
    list_all[id_list]="prop%d"%id_list
    id_list += 1
csvwriter.writerow(list_all)
model_list = {"Q.plant": 0, "Q.bird": 1, "Q.yeast": 2,
              "Q.mammal": 3, "Q.insect": 4, "Q.pfam": 5, "LG": 6, "WAG": 7, "JTT": 8}


def common_accestor(seq1, seq2, seq3):
    com_seq = ['-'] * len(seq1)
    for i in range(0, len(seq1)):
        com_i = '-'
        if seq1[i] == seq2[i] or seq1[i] == seq3[i]:
            com_i = seq1[i]
        elif seq2[i] == seq3[i]:
            com_i = seq2[i]
        com_seq[i] = com_i

    return com_seq


def process_phyml_file(input_file):
    lb = sys.argv[2]
    label = 0
    if "Q.plant" in input_file:
        label = 0
    if "Q.bird" in input_file:
        label = 1
    if "Q.yeast" in input_file:
        label = 2
    if "Q.mammal" in input_file:
        label = 3
    if "Q.insect" in input_file:
        label = 4
    if "Q.pfam" in input_file:
        label = 5
    if "LG" in input_file:
        label = 6
    if "WAG" in input_file:
        label = 7
    if "JTT" in input_file:
        label = 8
    in_file = open(input_file, "r")
    first_line = in_file.readline()
    taxa_count = int(first_line.split()[0])
    site_count = int(first_line.split()[1])
    data = [""*taxa_count for i in range(taxa_count)]
    count = 0
    for line in in_file:
        if len(line) > 10:
            line = line.split()[1]
            data[count] = line
            count += 1
    out_matrix = [[0]*20 for i in range(20)]
    out_matrix_triplet = [[0]*20 for i in range(20)]
    sum = 0
    rand_arr = [0] * 20
    loop = max(int(400000/site_count), taxa_count)
    for id in range(loop):
        rand1 = random.randint(0, taxa_count-1)
        rand2 = random.randint(0, taxa_count-1)

        while(rand2 == rand1):
            rand2 = random.randint(0, taxa_count-1)

        rand3 = random.randint(0, taxa_count - 1)
        while (rand3 == rand1) or (rand3 == rand2):
            rand3 = random.randint(0, taxa_count - 1)

        line1 = list(data[rand1])
        line2 = list(data[rand2])
        line3 = list(data[rand3])
        common_ans = common_accestor(line1, line2, line3)
        for id_i in range(site_count):
            if (line1[id_i] in list_aa) and (line2[id_i] in list_aa):
                id1 = dict_aa[line1[id_i]]
                id2 = dict_aa[line2[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
            if (line1[id_i] in list_aa) and (line3[id_i] in list_aa):
                id1 = dict_aa[line1[id_i]]
                id2 = dict_aa[line3[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
            if (line2[id_i] in list_aa) and (line3[id_i] in list_aa):
                id1 = dict_aa[line2[id_i]]
                id2 = dict_aa[line3[id_i]]
                out_matrix[id1][id2] += 1
                sum += 1
        for id_i in range(site_count):
            if common_ans[id_i] in list_aa:
                if line1[id_i] in dict_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line1[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
                if line2[id_i] in list_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line2[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
                if line3[id_i] in list_aa:
                    id1 = dict_aa[common_ans[id_i]]
                    id2 = dict_aa[line3[id_i]]
                    out_matrix_triplet[id1][id2] += 1
                    sum += 1
    data = [0]*801
    data_c = 1
    data[0] = label

    for i in range(20):
        for j in range(20):
            key = list_aa[i] + list_aa[j]
            dict_all[key] = out_matrix[i][j]
            data[data_c] = out_matrix[i][j]
            data_c += 1
    for i in range(20):
        for j in range(20):
            key = list_aa[i] + list_aa[j]
            dict_all[key] = out_matrix_triplet[i][j]
            data[data_c] = out_matrix_triplet[i][j]
            data_c += 1
    csvwriter.writerow(data)


for f in glob.glob("%s/*" % folder):
    #print("Proess file %s" % f)
    process_phyml_file(f)
