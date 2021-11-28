import nltk
import sys
import os
import string
import math
import itertools
import re

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf8") as f:
            try:
                files[file] = f.read()
            except Exception as e:
                print(e)
    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = nltk.tokenize.word_tokenize(document.lower())
    stopwords = nltk.corpus.stopwords.words("english")
    punctuation = re.compile(f'[{string.punctuation}]')

    return list(
        word for word in words 
        if punctuation.match(word) is None and word not in stopwords
    )


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    # check each word for frequency
    idfs = {}
    for document in documents:
        # all the words in the current document, not repeated
        words = list(dict.fromkeys(documents[document]))
        for word in words:
            if word in idfs:
                idfs[word] += 1
            else:
                idfs[word] = 1
    # calculate idfs
    for word in idfs:
        idfs[word] = math.log(len(documents) / idfs[word])
    
    return idfs
    

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # match filename with its "score"
    scores = {}
    for file in files:
        scores[file] = 0
        for word in query:
            # check if word is in document
            if word not in files[file]:
                continue
            # check word frequency in file, add tf-idf to total score
            freq = files[file].count(word)
            scores[file] += freq * idfs[word]
    # order by score, descending
    best = list(scores.keys())
    best.sort(reverse=True, key=lambda x: scores[x])

    # return 'n' top files
    return best[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    scores = {}
    for sentence in sentences:
        # [word idf, word density]
        scores[sentence] = [0, 0]
        for word in query:
            # sum of idfs and word density
            if word in sentences[sentence]:
                scores[sentence][0] += idfs[word]
                scores[sentence][1] += sentence.lower().count(word) / len(sentences[sentence])
    best = list(scores.keys())
    best.sort(reverse=True, key=lambda x: (scores[x][0], scores[x][1]))

    return best[:n]
    

if __name__ == "__main__":
    main()
