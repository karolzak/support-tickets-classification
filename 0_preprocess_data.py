import os
from sklearn import preprocessing
import sys
import numpy as np
import pandas as pd
import pickle
from azureml.dataprep import package
sys.path.append(".")
sys.path.append("..")

removedWordsList = (['xxxxx1'])


def removeNonEnglish(text, englishWords):
    global removedWordsList
    wordList = text.split()
    if len(wordList) == 0:
        return " "
    y = np.array(wordList)
    x = np.array(englishWords)
    index = np.arange(len(englishWords))
    sorted_index = np.searchsorted(x, y)
    yindex = np.take(index, sorted_index, mode="clip")
    mask = x[yindex] != y
    maskedArr = np.ma.array(yindex, mask=mask).compressed()
    result = x[maskedArr]
    text = np.array2string(result)\
        .replace("\'", "")\
        .replace("[", "")\
        .replace("]", "")\
        .replace("\n", "")\
        .replace("\r", "")

    # Logging removed words
    removedWords = set(wordList)-set(result)
    removedWordsList += set(list(removedWords))-set(removedWordsList)
    return text


def encryptSingleColumn(data):
    le = preprocessing.LabelEncoder()
    le.fit(data)
    return le.transform(data)


def encryptColumnsCollection(data, columnsToEncrypt):
    for column in columnsToEncrypt:
        data[column] = encryptSingleColumn(data[column])
    return data


def removeString(data, regex):
    return data.str.lower().str.replace(regex.lower(), ' ')


def cleanDataset(dataset, columnsToClean, regexList):
    for column in columnsToClean:
        for regex in regexList:
            dataset[column] = removeString(dataset[column], regex)
    return dataset


def getRegexList():
    regexList = []
    regexList += ['From:(.*)\r\n']  # from line
    # regexList += ['RITM[0-9]*'] # request id
    # regexList += ['INC[0-9]*'] # incident id
    # regexList += ['TKT[0-9]*'] # ticket id
    regexList += ['Sent:(.*)\r\n']  # sent to line
    regexList += ['Received:(.*)\r\n']  # received data line
    regexList += ['To:(.*)\r\n']  # to line
    regexList += ['CC:(.*)\r\n']  # cc line
    regexList += ['The information(.*)infection']  # footer
    regexList += ['Endava Limited is a company(.*)or omissions']  # footer
    regexList += ['The information in this email is confidential and may be legally(.*)interference if you are not the intended recipient']  # footer
    regexList += ['\[cid:(.*)]']  # images cid
    regexList += ['https?:[^\]\n\r]+']  # https & http
    regexList += ['Subject:']
    # regexList += ['[\w\d\-\_\.]+@[\w\d\-\_\.]+']  # emails
    # regexList += ['[0-9][\-0–90-9 ]+']  # phones
    # regexList += ['[0-9]']  # numbers
    # regexList += ['[^a-zA-z 0-9]+']  # anything that is not a letter
    # regexList += ['[\r\n]']  # \r\n
    # regexList += [' [a-zA-Z] ']  # single letters
    # regexList += [' [a-zA-Z][a-zA-Z] ']  # two-letter words
    # regexList += ["  "]  # double spaces

    regexList += ['^[_a-z0-9-]+(\.[_a-z0-9-]+)*@[a-z0-9-]+(\.[a-z0-9-]+)*(\.[a-z]{2,4})$']
    regexList += ['[\w\d\-\_\.]+ @ [\w\d\-\_\.]+']
    regexList += ['Subject:']
    regexList += ['[^a-zA-Z]']

    return regexList


if __name__ == '__main__':
    ####################
    # Use this with AML Workbench to load data from data prep file
    # dfIncidents = package.run('Incidents.dprep', dataflow_idx=0)
    # dfIncidents = pd.read_csv('allIncidents.csv', encoding="ISO-8859-1")
    # dfRequests = package.run('Requests.dprep', dataflow_idx=0)
    dfIncidents = package.run('IncidentsCleaned.dprep', dataflow_idx=0)
    dfRequests = package.run('RequestsCleaned.dprep', dataflow_idx=0)

    # Load dataset from file
    # dfIncidents = pd.read_csv('./data/endava_tickets/all_incidents.csv')
    # dfRequests = pd.read_csv('./data/endava_tickets/all_requests.csv')
    #####################

    # Reorder columns
    columnsOrder = [
        'title', 'body', 'ticket_type', 'category',
        'sub_category1', 'sub_category2', 'business_service',
        'urgency', 'impact'
    ]
    dfIncidents = dfIncidents[columnsOrder]
    dfRequests = dfRequests[columnsOrder]
    print(dfIncidents.shape)
    print(dfRequests.shape)

    # Merge incidents and requests datasets
    dfTickets = dfRequests.append(
        dfIncidents,
        ignore_index=True)  # set True to avoid index duplicates
    print(dfTickets.shape)

    # Remove duplicates
    columnsToDropDuplicates = ['body']
    dfTickets = dfTickets.drop_duplicates(columnsToDropDuplicates)
    print(dfTickets.shape)

    # Merge 'title' and 'body' columns into single column 'body'
    # dfTickets['body'] = (dfTickets['title']+
    #   " " + dfTickets['body']).map(str)
    # dfTickets = dfTickets.drop(['title'], axis=1)

    # Select columns for cleaning
    columnsToClean = ['body', 'title']

    # Create list of regex to remove sensitive data
    # Clean dataset and remove sensitive data
    cleanDataset(dfTickets, columnsToClean, getRegexList())

    ########################################
    # Remove all non english words + names #
    ########################################
    # Firstly load english words dataset and names dataset
    # dfWordsEn = package.run('EnglishWords.dprep', dataflow_idx=0)
    # dfWordsEn = package.run('EnglishWordsAlpha.dprep', dataflow_idx=0)
    # dfWordsEn = package.run('EnglishWordsMerged.dprep', dataflow_idx=0)
    dfWordsEn = package.run('WordsEn.dprep', dataflow_idx=0)
    dfFirstNames = package.run('FirstNames.dprep', dataflow_idx=0)
    dfBlackListWords = package.run('WordsBlacklist.dprep', dataflow_idx=0)

    # Transform all words to lower case
    dfWordsEn['Line'] = dfWordsEn['Line'].str.lower()
    dfFirstNames['Line'] = dfFirstNames['Line'].str.lower()
    dfBlackListWords['Line'] = dfBlackListWords['Line'].str.lower()

    # Merge datasets removing names from English words dataset
    print("Shape before removing first names from\
        english words dataset: "+str(dfWordsEn.shape))
    dfWords = dfWordsEn.merge(
        dfFirstNames.drop_duplicates(),
        on=['Line'], how='left', indicator=True)
    # Select words without names only
    dfWords = dfWords.loc[dfWords['_merge'] == 'left_only']
    print("Shape after removing first names from\
        english words dataset: "+str(dfWords.shape))
    dfWords = dfWords.drop("_merge", axis=1)  # Drop merge indicator column

    # Merge datasets removing blacklisted words
    print("Shape before removing blacklisted\
        words from english ords dataset: "+str(dfWords.shape))
    dfWords = dfWords.merge(
        dfBlackListWords.drop_duplicates(),
        on=['Line'], how='left', indicator=True
    )
    # Select words
    dfWords = dfWords.loc[dfWords['_merge'] == 'left_only']
    print("Shape after removing blacklisted\
        words from english words dataset: "+str(dfWords.shape))

    # Remove non english words and names
    dfTickets['body'] = dfTickets['body'].apply(
        lambda emailBody: removeNonEnglish(emailBody, dfWords['Line']))
    dfTickets['title'] = dfTickets['title'].apply(
        lambda emailBody: removeNonEnglish(emailBody, dfWords['Line']))

    # Remove empty strings and null rows after removing non english words
    print("Before removing empty: " + str(dfTickets.shape))
    dfTickets = dfTickets[dfTickets.body != " "]
    dfTickets = dfTickets[dfTickets.body != ""]
    dfTickets = dfTickets[~dfTickets.body.isnull()]
    print("After removing empty: " + str(dfTickets.shape))

    ########################################################
    # Data encryption and anonymization using LabelEncoder #
    ########################################################
    # Select columns for encryption
    columnsToEncrypt = [
        'category', 'sub_category1', 'sub_category2',
        'business_service', 'urgency',
        'impact', 'ticket_type'
    ]

    # Encrypt data for each of selected columns
    dfTickets = encryptColumnsCollection(dfTickets, columnsToEncrypt)

    ##########################

    # Remove duplicates x2
    columnsToDropDuplicates = ['body']
    dfTickets = dfTickets.drop_duplicates(columnsToDropDuplicates)
    print(dfTickets.shape)

    # Save cleaned and encrypted dataset back to csv without indexes
    dfTickets.to_csv('all_tickets.csv', index=False, index_label=False)
    sortedRemovedWordsList = np.sort(removedWordsList)
    dfx = pd.DataFrame(sortedRemovedWordsList)
    dfx.to_csv("removed_words.csv", index=False, index_label=False)
