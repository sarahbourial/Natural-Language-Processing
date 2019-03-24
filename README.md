NLP Exercise 2: Aspect Based Sentiment Analysis

Student who took part in this assignment:

-     Meryem BEN-GOUMI <meryem.ben-goumi@student-ecp.fr>
-     Sarah BOURIAL <sarah.bourial@student-cs.fr>
-     Gabrielle RAPPAPORT <gabrielle.rappaport@student-ecp.fr>
-     Josiak KAYO-KOUAKAM <josias.kayo-kouakam@student-ecp.fr>

Description of our delivrable

	a) Feature representation
  To prepare our dataset for training, we first got rid of all punctuations and turned all words into lower case. We then extracted the 
  category terms that we use to train a categories classifier.  
  
  We encoded our category terms as vectors to feed them into our category training model using a BOW method after turning them into to 
  dummy variables (as as we cannot include them as strings into the model. 
  
  We then extracted polarity terms (essentially the words translating the main sentiment of the review) and computed the distance 
  between remanings words in the sentences and our main sentiment term. We used spaCy dependency for a Part Of Speech (POS) Tagging 
  to filter and keep only adjectives and verbs which we assumed to best translate the connotation of the review sentiment. The same 
  BOW encoding as before was used here to feed our model with this data
  
  
  b) Classifier
  Our final ABSA implementation was done with a 1D Convolutional Neural Network 
  
  c) Results:
  Our best accuracy was achieved with a Convolutional 1D model which led to an accuracy of 0.75.
  We also trained an LSTM on our classifier, however the results were less satisfying generating an accuracy of 0.70 only. 
  An interesting extension of our work would be attempting to train our model with an svm SVC model which we believe could lead to 
  an increased accuracy. Another extension would be computing a Vader polarizer score (it did not seem to be allowed for this assignment
  on the reviews before feeding the data into our model, of which probabilities should lead to better predictions on the 'neutral' reviews
  which was the polarity level that was the least well predicted by our model. 
