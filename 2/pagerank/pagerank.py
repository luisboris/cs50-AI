import os
import random
import numpy
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.
    """
    model = {}

    # page has no links
    if len(corpus[page]) == 0:
        for link in corpus:
            model[link] = 1 / len(corpus)
    else:
        # probability for random link from corpus
        for link in corpus:
            model[link] = (1 - damping_factor) / len(corpus)
            
        # probability for random link from page
        for link in corpus[page]:          
            model[link] += damping_factor / (len(corpus[page]))

    return model


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.
    """
    link = random.choice(list(corpus))

    # number of times each page was visited
    clicks = {}
    for page in corpus:
        clicks[page] = 0

    # generate samples by clicking links at random following distribution code
    for i in range(n):
        distribution = transition_model(corpus, link, damping_factor)
        link = numpy.random.choice(list(corpus), p=list(distribution.values()))
        clicks[link] += 1
    
    # calculate PageRank of each page
    ranks = {}
    for page in corpus:
        ranks[page] = clicks[page] / n

    return ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # define initial rank, equal for every page
    ranks = {}
    for page in corpus:
        ranks[page] = 1 / len(corpus)
    
    # iterate until convergence in all PageRanks
    while True:
        
        previous_ranks = copy.deepcopy(ranks)
        convergence = 0
        for page in corpus:

            # get group of pages that link to current page 
            # (if page has no links, behave as if having one link for every page - including itself)
            links = {}
            for link in corpus:
                if len(corpus[link]) == 0 or page in corpus[link]:
                    links[link] = link
            
            # get PageRank
            sum = 0
            for i in links:
                sum += ranks[i] / len(corpus[i])
            ranks[page] = ((1 - damping_factor) / len(corpus)) + (damping_factor * sum)
            
            # check for convergence
            if abs(ranks[page] - previous_ranks[page]) < 0.001:
                convergence += 1
        
        if convergence == len(corpus):
            break
    
    return ranks



if __name__ == "__main__":
    main()
