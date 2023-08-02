def no_of_subsequences(a, k):
    seq_no = 0

    no = 0

    arr = sorted(a, key=lambda x: x * x)

    i = 1

    while i < n:
        seq_no = i + 1

        i += k

    if seq_no == 0:
        seq_no = n

    return seq_no
assert no_of_subsequences([1,2,3,4], 10) == 11
assert no_of_subsequences([4,8,7,2], 50) == 9
assert no_of_subsequences([5,6,7,8], 15) == 4
