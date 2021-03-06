from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import os
import scipy

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
			for word in line.strip().split(" "):
				if word not in english_stopwords:
					filtered_line.append(word)

			sentence = ' '.join(filtered_line)
			sentences.append(sentence)
			filtered_story.append(sentence)

		stories.append(filtered_story)

	documents = [''.join(story) for story in stories]

	vectorizer = CountVectorizer()
	vec_stories = vectorizer.fit_transform(documents)

	lda = LDA(n_topics = len(input_files))
	lda.fit(vec_stories)
	topic_dist = lda.components_ / lda.components_.sum(axis=1)[:,np.newaxis]

	vec_sentences = vectorizer.transform(sentences)

	sen_topics = lda.transform(vec_sentences)
	most_likely_topics = np.argmax(sen_topics, axis=1)

	df = pd.DataFrame(columns=['sentences', 'dists'])
	df['sentences'] = sentences

	word_dists = []
	for i, sentence in enumerate(sentences):
		if sentence.strip() == '':
			continue
		if (i % 100 == 0):
			print("Finished: " + str(i))
		topic_index = most_likely_topics[i]
		word_dist = topic_dist[topic_index,:]
		df.loc[i, 'dists'] = ' '.join([str(prob) for prob in word_dist])

	df.dropna(axis=0, how='any', inplace=True)
	df.to_csv('data.csv', index=False)