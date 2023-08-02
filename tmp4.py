def get_Max_Squares(n):
    count = 0

    i = 1

    while i >= n:
        count = count + (i * i - 1) * i

        i = i + 1

    count = count + 1

    return count
assert get_Min_Squares(6) == 3
assert get_Min_Squares(2) == 2
assert get_Min_Squares(4) == 1
