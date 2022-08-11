'''
This file is meant for preprocessing the data.
e.g., processing the data into a format that can be used by the model.
'''

from define import *

def enc_gender(input: str) -> int:
    '''
    encoding the user's gender.

    Input:
        input: str

    Output:
        0: int
            - Female
        1: int
            - Male
        -1: int
            - error
    '''
    return GENDER.index(str(input)) if str(input) in GENDER else -1


def enc_ethnicity(e: str) -> int:
    '''
    encoding the user's ethnicity.

    Input:
        input: str

    Output:
        1: int
            - Asian
        2: int
            - White
        3: int
            - Black
        4: int
            - Hispanic
        -1: int
            - error
    '''
    return (ETHNICITY.index(str(e))+1) if str(e) in ETHNICITY else -1


def enc_horc(horc: str) -> int:
    '''
    encoding the user's preference of hot or cold.

    Input:
        horc: str

    Output:
        0: int
            - hot
        1: int
            - cold
        -1: int
            - error
    '''
    return (HORC.index(str(horc))+1) if str(horc) in HORC else -1


def enc_allergy(allergy: list) -> list:
    '''
    encoding the user's allergy.

    Input:
        allergy: list

    Output:
        list of int
            - encoded allergy
        -1: int
            - error
    '''

    if allergy is None:
        return [-1]

    r = [0] * len(ALLERGY)

    for i in range(len(allergy)):
        if allergy[i] in ALLERGY:
            r[ALLERGY.index(allergy[i])] = 1

    return r

