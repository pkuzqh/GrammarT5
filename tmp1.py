def cal_electbill(n):
    if n * (math.pi * 2) == 0:
        return 0

    cal_electbill = (n * (math.pi * 2 + math.pi * 2)) / 2

    return cal_electbill
assert cal_electbill(75)==246.25
assert cal_electbill(265)==1442.75
assert cal_electbill(100)==327.5
