# NLP - Skip-Gram with Negative Sampling

This script is an implementation of Skip-Gram with Negative Sampling, that follows the approach explained by Adrien Guille's in a post detailing Mikolov et al's method. It consists of the optimisation of a log-likelihood function using a stochastic gradient descent. 

This document is meant to help you go through the different functions implemented and understand their output. However, the logic behind each implementation as well as mathematical demonstrations are detailed in our report.


## Initialization

The funtion **init** of the class mSkipGram relies on the four steps explained below: 

### Generating a set of Sentences

After reading the path, the first step is to transform the input into an appropriate set of sentences:
- The function **text2sentences** converts a raw text into tokenized sentences, splitting the input at each punctuation that ends a sentence i.e. <.>, <!> and <?> and subsequently deleting those punctuations.
- The function **pre_initialisation** creates dictionaries that count the occurences of words and assign them to unique indices. It also removes words that occur less than minCount (rare words).

The output of this step is a set of tokenized sentences with no rare words. 

### Undersampling (or not undersampling)

The function **undersample_sentences** was implemented to avoid using a stopwords list. The idea is to use a weighting function to try to undersample very frequent wods. We have chosen the threshold of the undersampling after testing different values in [0.9,0.95] and plotting the probability distribution of words in our sample. 

We have decided to not use the undersampling for our final results, as it does not improve the performance with regards to our hyperparameters. However, in case we wanted to try it again, we can simply uncomment line 65 and comment line 66 (return lines of the function **text2sentences**).

### Generating Positive pairs of words

The function **targetw_contextw_positive_pairs** generates triplets of positive words: (target word, context word, +1). For each word in each sentence, the context words are found with sliding a window of size winSize centered in the target word. 

### Generating Negative Pairs of words

The function **targetw_contextw_negative_pairs** generates triplets of negative words: (target word, context word, -1). To do so, it creates an Adaptative Unigram Table, and then randomly choses k (negativeRate) context words from this table. 

## Training

This step takes as input the set of triplets generated from the initialization, and computes the matrices U and V that maximize the log-likelihood between target words and context words. Each row of matrix U (resp. V) is the vectorial representation of the corresponding target word (resp. context word).

### Initializing Embedding Matrices

The training process starts with a random initialization of U and V.

### Updating Embedding Matrices with Stochastic Gradient Descent

At each epoch, we randmly shuffle our set of triplets and split it into different batches. Going through each batch, every triplet of (target word, context word) encountered leads to an update of the corresponding rows in matrices U and V. The new rows are computed using a Stochastic Gradient Descent with a constant step size. 
Hence, the function computes the gradient of the log-likelihood and uses it to update matrices U and V. 

### Output of the Training phase

When launched, the script prints information to keep track of its evolution: 

```
Parameters:  #nEmbed: 100, negativeRate: 10, Window Size:5, minCount:5

Starting Initialization: 2019-02-21 20:46:57.769803

Pre-Processing

Generating Positive Pairs: : 2019-02-21 20:46:57.774816

Generating Negative pairs through Unigram Distribution: : 2019-02-21 20:46:57.936679

Initialisation is Done: : 2019-02-21 20:47:21.029256

TRAINING: #epochs: 5, step size: 0.025, batch size:1024

Epoch n� : 1/5 - 2019-02-21 20:47:21.556861

Epoch n� : 2/5 - 2019-02-21 21:03:45.884798

Epoch n� : 3/5 - 2019-02-21 21:19:31.343887

Epoch n� : 4/5 - 2019-02-21 21:35:19.570179

Epoch n� : 5/5 - 2019-02-21 21:51:01.848101

Training ended

We are saving the results
```

The main output is the matrix of embeddings U. Once the training is done, we launch commands similar to the one below, to check if our model is able to find the closest words to our input word: 

```
find_most_similar(sg.U, 5, "president", sg.target_words_dict, sg.target_words)
```

The function **find_most_similar** computes similarities (based on cosine distance) between the word "president" and other words in the vocabulary, sorts them, and returns the 5 most similar ones. The output of the command above after training the model with 1000 sentences is: 

```
In [166]: find_most_similar(sg.U, 5, "president", sg.target_words_dict, sg.target_words)

The 5 closest words are president:

(0.40824829046386296, 'years')

(0.40824829046386296, 'obama')

(0.35511041211421746, 'known')

(0.3171207896914326, 'cost')

(0.31622776601683794, 'then')
```

### Running the Model: Default Parameters

After comparing the results, we have set the following default values:
- Window Size: 5 
- negativeRate: 5 
- Epochs: 10
- Step Size: 0.025
- Batch Size: 512

As it is explained in our report, those default parameters are the ones that resulted in the best performance, except for negativeRate. Indeed, we obtain the best results for a negativeRate = 10, since we have only been able to run the model with a small set of sentences (up to 5000 sentences). Still, we have set the default value as 5 because this script is meant to be run on 100 000 sentences in less than 48 hours. Besides, the references mentioned in the introduction claim that a negativeRate of 5 is suitable to large datasets.

With the default parameters above, the model should run in 20 hours (roughly 2 hours per epoch). 
