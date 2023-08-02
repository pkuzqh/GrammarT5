import re

regex = "are"


def multiple_split(string):
    return re.findall(regex, string)
assert multiple_split('Forces of the \ndarkness*are coming into the play.') == ['Forces of the ', 'darkness', 'are coming into the play.']
assert multiple_split('Mi Box runs on the \n Latest android*which has google assistance and chromecast.') == ['Mi Box runs on the ', ' Latest android', 'which has google assistance and chromecast.']
assert multiple_split('Certain services\nare subjected to change*over the seperate subscriptions.') == ['Certain services', 'are subjected to change', 'over the seperate subscriptions.']
