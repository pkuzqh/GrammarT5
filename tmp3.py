from sys import maxsize


def split_two_parts(list1):
    parts = list(filter(lambda ele: (ele[0] > ele[1]), list1))

    return parts
assert split_two_parts([1,1,2,3,4,4,5,1],3)==([1, 1, 2], [3, 4, 4, 5, 1])
assert split_two_parts(['a', 'b', 'c', 'd'],2)==(['a', 'b'], ['c', 'd'])
assert split_two_parts(['p', 'y', 't', 'h', 'o', 'n'],4)==(['p', 'y', 't', 'h'], ['o', 'n'])
