# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 16:23:10 2019

@author: Meryem
"""

#from _future_ import division

import argparse
import pickle
from math import ceil, sqrt

# useful stuff
import numpy as np
import pandas as pd

from numpy.random import choice
from scipy.special import expit
import regex as re 
import string
import datetime

import matplotlib.pyplot as plt
from collections import Counter

_authors_ = ['Bourial Sarah', 'Rappaport Gabrielle', 'Ben-Goumi Meryem', 'Josias Kayo Kouokam']

_emails_ = ['sarah.bourial@student-cs.fr', 'gabrielle.rappaport@student.ecp.fr', 'meryem.ben-goumi@student.ecp.fr', 'josias.kayo-kouokam@student.ecp.fr']

''' ________________________ PREPARING DATA SET ________________________ '''

def text_concat(path):
    '''
    Concatenates all the text in a file located at file path
    '''
    texts_list = []
    with open(path, encoding='utf8') as f:
        for l in f:
            l = l[:-1] #get rid of line jumping
            texts_list.append(l)
    text_concat = ''.join(texts_list)
    return text_concat

def text2sentences(path):
    '''
    Converts a raw text from path to tokenized sentences 
    '''
    concat_text = text_concat(path)
    tok_sentences = []
    # Split at each symbol that ends a sentence e.g. '.?!'
    sents = re.split(r'[.?!]', concat_text)
    
    # Get rid of punctuation
    punct = re.compile('[%s]' % re.escape(string.punctuation))
    # Tokenize
    for phrase in sents:
        no_punct = punct.sub('', phrase)
        tok_sentences.append(no_punct.lower().split() )

    #return undersample_sentences(tok_sentences)
    return tok_sentences #comment this line & uncomment the one above if we want to use the undersampling
    
def undersample_sentences(tok_sentences):
    '''
    Undersamples very frequent words in our corpus of sentences
    '''
    print("Undersample starts")
    tok_sentences = tok_sentences
    count_words_all, count_words, index_words, sentences_with_minCount = pre_initialization(tok_sentences)
    proba = []
    proba_threshold = 0.93  # Threshold chosen  after plotting the graph (cf below), then comparing lists of words_to_be_undersampled for threshold [0.9, 0.95].
    words_to_be_undersampled = []
    t = 10 ** (-5)
    total_number_words = sum(count_words_all.values())
    for word in index_words:
        freq = count_words_all[word] / total_number_words
        p = 1 - sqrt(t / freq)
        proba.append(p)
        if p > proba_threshold:
            words_to_be_undersampled.append(word)

    # Undersample
    sentences = tok_sentences

    for sent in sentences:
        for word in sent:
            if word in words_to_be_undersampled:
                erase = np.random.choice(2, 1, p=[0.99, 0.01])  #We need to find a definition for this probability
                if erase:
                    sent.remove(word)

    print("words_to_be_undersampled: " + str(words_to_be_undersampled))
    print(len(words_to_be_undersampled))
    # Plot

    c = Counter(proba)
    print(c)
    print(max(c.values()))
    print(max(c.keys()))
    
    #Plotting the probability distribution of words in our sample
    plt.plot(c.keys(), c.values(), 'ro')
    plt.show()

    return sentences

def pre_initialization(sentences, minCount=5):
    """
    Computes the dictionaries that count the occurences of words and allow us to track the indices,
    And removes words that occur less than minCount (rare words)
    :param sentences: set of sentences
    :param minCount: minimum number of occurences to keep a word in the dataset
    :return: 
        count_words_all : dictionary with number of occurences of all words
        count_words : dictionary with number of occurences of words occuring more than minCount
        index_words : list of unique words (helps to associate each word to a unique index)
        sentences_with_minCount : new list of sentences, without rare words
    """
    # Dictionary that will contain number of Occurences of each word
    count_words_all = {}  # for all words
    count_words = {}  # for only words than occure more than minCount

    # List that associates each word to a unique index (its position in the list)
    index_words = []

    print("Cleaning the Set of sentences")

    #Cleaning: Keeping only words that appear more than minCount

    # Compute the number of Occurences of each word 
    for sent in sentences:
        for word in sent:
            if word in count_words_all:
                count_words_all[word] += 1
            else:
                count_words_all[word] = 1

    # Keeping only words that appear more than minCount
    # And assigning them a unique index with index_words
    sentences_with_minCount = []
    for sent in sentences:
        sent_with_minCount = []
        for word in sent:
            if count_words_all[word] >= minCount:
                sent_with_minCount.append(word)
                if word not in count_words:
                    count_words[word] = count_words_all[word]
                    index_words.append(word)

        sentences_with_minCount.append(sent_with_minCount)

    return count_words_all, count_words, index_words, sentences_with_minCount


def loadPairs(path):
    data = pd.read_csv(path, delimiter='\t')
    pairs = zip(data['word1'], data['word2'], data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize=5, minCount=5):
        """
        Initialisation Step: Generates the triplets of (target word, context word, +/- 1)
        """
            
        print("Parameters:  #nEmbed: {}, negativeRate: {}, Window Size:{}, minCount:{}".format(nEmbed,negativeRate,winSize,minCount))
        print("Starting Initialization: {}".format(str(datetime.datetime.now())))
        
        self.sentences = sentences
        self.nEmbed = nEmbed
        self.negativeRate = negativeRate
        self.winSize = winSize
        self.minCount = minCount

        self.count_words_all, self.count_words, self.index_words, sentences_with_minCount = pre_initialization(
            self.sentences, self.minCount)

        print("Generating Positive Pairs: : {}".format(str(datetime.datetime.now())))

        # Generate positive pairs: list of (target word, context word, +1) pairs
        self.positive_pairs, self.context_words_positive, self.target_words_positive_list, self.target_words_positive_dict = self.targetw_contextw_positive_pairs(sentences_with_minCount,
                                                                                                winSize)

        print("Generating Negative pairs through Unigram Distribution: : {}".format(str(datetime.datetime.now())))

        # Negative Sampling
        # Generate negative pairs: list of (target word, context word, -1) pairs

        self.negative_pairs, self.context_words, self.target_words, self.target_words_dict = self.targetw_contextw_negative_pairs(self.count_words,
                                                                                       self.index_words,
                                                                                       self.positive_pairs,
                                                                                       self.context_words_positive,
                                                                                       self.target_words_positive_list,
                                                                                       self.target_words_positive_dict,
                                                                                       negativeRate)

        self.pairs = self.positive_pairs + self.negative_pairs

        print("Initialisation is Done: : {}".format(str(datetime.datetime.now())))

    def targetw_contextw_positive_pairs(self, sentences, winSize):
        """
        Creates the dataset with (target word, context word, +1) pairs 
        :param sentences: set of sentences
        :param winSize: size of the sliding window
        
        :return: 
            pairs: list of (target word, context word, +1) triplets
            context_words_positive: list of unique positive context words
            target_words_positive_list: list of unique positive target words
            target_words_positive_dict: dictionary target_word:index
        """
        pairs = []
        context_words_positive = []
        target_words_positive_list = []
        target_words_positive_dict = {} #dictionnary word:index

        target_w_index = 0
        
        for sent in sentences:
            counter = 0
            for target_word in sent:
                for context_w_index in range(max(0, counter - winSize),
                                             min(counter + winSize + 1, len(sent))):
                    context_word = sent[context_w_index]
                    if context_word != target_word:
                        pairs.append((target_word, context_word, +1))  # +1 because positive pair
                        if context_word not in context_words_positive:
                            context_words_positive.append(context_word)
                counter += 1
                            
                if target_word not in target_words_positive_list:
                    target_words_positive_list.append(target_word)
                    target_words_positive_dict[target_word] = target_w_index
                    target_w_index += 1

        return pairs, context_words_positive, target_words_positive_list, target_words_positive_dict

    def targetw_contextw_negative_pairs(self, count_words, index_words, positive_pairs, context_words_positive, 
                                        target_words_positive_list,
                                        target_words_positive_dict,
                                        negativeRate=5):
        """
        Creates the dataset with (target word, context word, -1) triplets (negative pairs of words) 
        :param count_words: dictionary with the number of occurences of words
        :param index_words: list of unique words
        :param positive_pairs: list of positive triplets (target word, wontext word, +1)
        :param context_words_positive: list unique context words that are in the set of positive pairs
        :param target_words_positive_list: list of unique target words that are in the set of positive pairs
        :param target_words_positive_dict: dictionary target_word:index
        :param negativeRate: ratio of number of negative pairs to number of positive pairs
        
        :return: 
            negative_pairs: list of (target word, context word, -1) triplets
            context_words_total: list of total context words (positive and negative)
            target_words_total_list: list of total target words (positive and negative)
            target_words_total_dict: dictionary target_word:index
        """
        # Create a Unigram Table: the nbre of times a word's index appears in the table
        # is given by P(w_i)*unigram_table_size
        unigram_table = []

        # Set a size for the table to avoid complex time computation
        unigram_table_max_size = len(index_words)

        # we use for "proba" the definition of probability P explained in section 4.1 of the report
        current_index = 0
        unigram_table_current_size = 0

        number_distinct_words = len(index_words)

        # Denominator of the probability P(w_i)
        Proba_denominator = float(sum([count_words[word] ** (3 / 4) for word in index_words]))

        while current_index < number_distinct_words and unigram_table_current_size < unigram_table_max_size:
            
            # probability associated to the word which index is "current_index"
            proba = count_words[index_words[current_index]] ** (3 / 4) / Proba_denominator

            F = proba * unigram_table_max_size
            # The expected number of times that the word appears should be F
            # As F is not an integer: we use int(F) or ceil(F) such as the expected nbre n is F

            elements = [int(F), ceil(F)]
            weights = [F - int(F), 1 - F + int(F)]
            n = choice(elements, p=weights)

            for k in range(n):
                unigram_table.append(current_index)
                unigram_table_current_size += 1
            current_index += 1

        # now let's build negative pairs
        negative_pairs = []

        # List of positive context words that we will complete with the negative ones
        context_words_total = context_words_positive
        target_words_total_list = target_words_positive_list
        target_words_total_dict = target_words_positive_dict
        
        target_word_index = len(target_words_total_dict)+1
        
        for positive_pair in positive_pairs:
            target_word = positive_pair[0]
            for k in range(negativeRate):
                random_index = choice(unigram_table)
                context_word = index_words[random_index]
                negative_pairs.append((target_word, context_word, -1))
                if context_word not in context_words_total:
                    context_words_total.append(context_word)
                    
            if target_word not in target_words_total_list:
                target_words_total_list.append(target_word)
                target_words_total_dict[target_word] = target_word_index
                target_word_index += 1

        return negative_pairs, context_words_total, target_words_total_list, target_words_total_dict

    def train(self, stepsize, epochs, batch_size):
        '''
        Using SGD - Steps followed from article: http://mediamining.univ-lyon2.fr/people/guille/word_embedding/skip_gram_with_negative_sampling.html?fbclid=IwAR3nvRj0S43AM0KuG7LGAOZ6t2xubxWolQfDnyllKP47BFrMSAn9dJrFsFk
        - Randomly initialise matrices of target words and context window
        - Iterate over dataset and update vectors according to SGD
        stepsize: the stochastic gradient descent
        epochs: nb of iterations through our training set:
        batchsize: nb of positive samples considered at each iteration of the gradient descent
        '''

        self.U = np.array([np.random.randint(low=0, high=100, size=self.nEmbed) for _ in
                           range(len(self.target_words))])  # vector of target words
        self.V = np.array([np.random.randint(low=0, high=100, size=self.nEmbed) for _ in
                           range(len(self.context_words))])  # vector of context words
        train_set = np.array(self.pairs)  # triplets set

        # Gradients
        # grad_U = np.array(sg.pairs)[:,2]*np.nditer(U)*self.sigmoid(-gamma*np.transpose.nditer(U)*V)
        # grad_V = np.array(sg.pairs)[:,2]*np.nditer(V)*self.sigmoid(-gamma*np.transpose.nditer(U)*V)

        # SGD
        print("TRAINING: #epochs: {}, step size: {}, batch size:{}".format(epochs, stepsize,batch_size))

        # Running through the epochs
        for epoch in range(epochs):
            print("Epoch n° : {}/{} - {}".format(epoch+1, epochs, str(datetime.datetime.now())))

            np.random.shuffle(train_set)
            
            train_set_size = len(train_set)
            
            for batch in range(int(train_set_size/batch_size)):
                    
                batch_first_idx = batch*batch_size
                batch_last_idx = min((batch+1)*batch_size, train_set_size)
                
                train_set_batch = train_set[batch_first_idx:batch_last_idx]
            
                for w_i, w_j, gamma_ij in train_set_batch:
                    gamma_ij = float(gamma_ij)
                    grad_ui = []
                    grad_vj = []
    
                    # index of words i and j
                    i = self.target_words.index(w_i)
                    j = self.context_words.index(w_j)
                
                        
                    for Lambda in range(self.nEmbed):
                        
                        product = gamma_ij * np.dot(self.U[i], self.V[j])
    
                        grad_ui_lambda = gamma_ij * self.V[j, Lambda] * expit(-product)  
                        grad_vj_lambda = gamma_ij * self.U[i, Lambda] * expit(-product)
    
                        grad_ui.append(grad_ui_lambda)
                        grad_vj.append(grad_vj_lambda)

                    self.U[i] = self.U[i] + stepsize * np.array(grad_ui)
                    self.V[j] = self.V[j] + stepsize * np.array(grad_vj)

        print("Training ended")

        pass

    def save(self, path):
        '''
        Save W (matrix of embeddings) and self.index_word (dictionnary of (word: index)) in path
        '''
        print("We are saving the results")
        with open(path, 'wb') as f:
            pickle.dump([self.U, self.target_words], f)

    def similarity(self, word1, word2):
        """
        Computes similiarity of words 1 and 2, using the output of the training phase
        :return: cosine distance between the embeddings of words 1 and 2 if they are in the vocabulary
        """
        
        #checking if word1 and word2 are in our target words' list
        if word1 in self.target_words and word2 in self.target_words:
            
            index1 = self.target_words_dict[word1]
            vec_1 = self.U[index1]
            
            index2 = self.target_words_dict[word2]
            vec_2 = self.U[index2]
            cosine_distance = vec_1.dot(vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
            return abs(cosine_distance)
            
        else:
            print("Error: word '{}' does not belong to the vocabulary".format(word1))

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            U, target_words = pickle.load(f)

        return U, target_words
    
def find_most_similar(U, n, word, target_words_dict, target_words_list):
    """
    Prints the n closest words to input word using the embeddings obtained fter the training set 
    :param U: matrix of embeddings of target words
    :param n: number of closest words 
    :param word: input word
    :param target_words_dict: dictionary target_word:index
    :param target_words_list: list of target_words in vocabulary
    """
    index_word = target_words_dict[word]
    nb_words = len(target_words_dict)
        
    similarities_list = []
    for k in range(nb_words):
        if k != index_word:
            #cosine similarity
            similarity = U[index_word].dot(U[k])/(np.linalg.norm(U[index_word])*np.linalg.norm(U[k]))
            similarities_list.append((similarity, target_words_list[k]))
                
    print("The {} closest words are {}:".format(n, word))
    #sort the list
    similarities_list = sorted(similarities_list, reverse=True)
    for i in range(n):
        print(similarities_list[i])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=False)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=False)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        path = r'C:\Users\Meryem\Desktop\OMA\NLP\1-billion-word\training-monolingual.tokenized.shuffled\news.en-00001-of-00100'
        sentences = text2sentences(path=path)[:2000]
        sg = SkipGram(sentences)
        sg.train(stepsize=0.025, epochs=10, batch_size=512)
        sg.save(r'C:\Users\Meryem\Desktop\OMA\NLP\result.txt')
        #find_most_similar(sg.U, 5, "school", sg.target_words_dict, sg.target_words)

    else:
        pairs = loadPairs(opts.text)
        U, target_words = SkipGram.load(opts.model)
        
        for a,b,_ in pairs:
            print(a, b, SkipGram.similarity(a,b,U,target_words))

