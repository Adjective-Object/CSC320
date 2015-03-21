from draw import draw
import numpy as np

def avg(l):
    return sum(l, 0.0) / len(l)

def extend_str(s, l):
    return s + " " * (l - len(s))

def pretty_table(table, num_format="{:.2f}"):
    table = [([" "+(item
                if isinstance(item, str)
                else (num_format.format(item)
                        if isinstance(item, float)
                        else str(item))) +" "
                for item in row]) if row else row
            for row in table]

    width = len(table[0])
    row_widths = [max([len(row[i]) for row in table if row]) for i in range(width)]


    hbar = "-" * sum(row_widths) + "-" * (len(row_widths) -1)

    def make_tabular(t):
        return ["|".join(
                map(lambda tup:
                        extend_str(tup[1], row_widths[tup[0]]),
                        enumerate(row))
                if row else [hbar]
                ) for row in t]


    return (" " + hbar + " " + "\n|" +
            "|\n|".join(
                make_tabular(table)
            ) + "|\n" +" " + hbar + " ")

def build_results_table(comps, gendermap):
    vals = []
    vals_gender = []
    restab = [["k", "Name", "Match", "Gender Match"]]
    for k in sorted(comps.keys()):
        restab.append(None)
        for name in comps[k].keys():
            matches = (
                len(filter(lambda n: n == name, comps[k][name])) * 1.0 /
                    len(comps[k][name]))

            gender_matches = (
                len(filter(lambda n: gendermap[n] == gendermap[name], 
                    comps[k][name])) * 1.0 / len(comps[k][name]))

            vals.append(matches)
            vals_gender.append(gender_matches)
            restab.append([
                k, name,
                "{:.2f}".format(matches),
                "{:.2f}".format(gender_matches)
            ])

        matches = avg(vals[- len(comps[k].keys()) + 1:])
        restab.append([
            "", "(avg)",
            "{:.2f}".format(matches),
            "{:.2f}".format(gender_matches)])
    restab.append(None)
    restab.append([
        "", "(avg of all)", 
        "{:.2f}".format(avg(vals)),
        "{:.2f}".format(avg(vals_gender))])
    return restab

def build_similarity_table(comps):
    simscores = {}
    for k in sorted(comps.keys()):
        for name in comps[k].keys():
            if not name in simscores:
                simscores[name] = {}
            for name_2 in comps[k].keys():
                if not name_2 in simscores[name]:
                    simscores[name][name_2] = 0
                simscores[name][name_2] += len(
                    filter(
                        lambda n: n == name_2,
                        comps[k][name]))

    header = [""] + comps[k].keys()

    # set up headers on table
    table = []
    for i in range(len(header)):
        table.append([ 0 ] * len(header))
    table[0] = map(lambda h: h.split("_")[0], header)

    for row, h in enumerate(header):
        table[row][0] = h

    # assign the values
    for i, key in list(enumerate(header))[1:]:
        vals = map(lambda name: simscores[key][name], header[1:])
        for x, val in enumerate(vals):
            table[i][x+1] = val

    return table[0:1] + [None] + table[1:]
