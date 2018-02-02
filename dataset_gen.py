from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os

DIR_PATH = './dm_stories_tokenized/'
english_stopwords = set(stopwords.words('english'))
N_TOP_WORDS = 100

if __name__ == "__main__":
	stories = []
	input_files = os.listdir(DIR_PATH)
	sentences = []

	for file in input_files:
		story = open(DIR_PATH + file, 'r', encoding='latin-1').readlines()
		filtered_story = []

		for line in story:
			filtered_line = []
			for word in line.split(" "):
				if word not in english_stopwords:
					filtered_line.append(word)

			sentence = ' '.join(filtered_line)
			sentences.append(sentence)
			filtered_story.append(sentence)

		stories.append(filtered_story)

	documents = [''.join(story) for story in stories]

	doc_vectorizer = CountVectorizer()
	vec_stories = doc_vectorizer.fit_transform(documents)

	lda = LDA(n_topics = len(input_files))
	lda.fit(vec_stories)

	sen_vectorizer = CountVectorizer()
	vec_sentences = sen_vectorizer.fit_transform(sentences)

	word_to_index = sen_vectorizer.vocabulary_
	index_to_word = np.chararray(len(word_to_index), itemsize=100)
	for word in word_to_index:
	    index = word_to_index[word]
	    index_to_word[index] = word

	sen_topics = lda.transform(vec_sentences)

	w_z = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

	top_word_args = np.argsort(w_z, axis=1)[:,-1*N_TOP_WORDS:]
	top_words = np.chararray((w_z.shape[0], N_TOP_WORDS, 2), itemsize=100)
	for i in range(len(input_files)):
	    top_words[i,:,0] = index_to_word[top_word_args[i]]
	    top_words[i,:,1] = w_z[i,top_word_args[i]]

	df = pd.DataFrame()
	df['sentences'] = pd.Series(sentences)
	for i in range(len(sen_topics.T)):
		df['topic-' + str(i)] = pd.Series(sen_topics.T[i])
	for i in range(len(top_words)):
		df['topic-' + str(i) + '-words'] = pd.Series(top_words[i][:,0])
		df['topic-' + str(i) + '-wordprobs'] = pd.Series(top_words[i][:,1])

	df.to_csv('data.csv', index=False)

