# Malicious-URL-DETECTION
A malicious URL is a clickable link implanted within the content of a dispatch. In
simple words, a malicious URL is a clickable link that directs users to a malicious or
otherwise fraudulent web page or website. As the name suggests, nothing good can
ever come out of a malicious URL. That’s because the goal of creating these bad site
pages is typically for a nefarious purpose — such as to carry out a political agenda,
steal personal or company data, or make a quick buck.

For the scope of this study, we propose a two-pronged feature
extraction method. In the first process, the URL is tokenized into an array of
ASCII values of the characters in the URL.Then, if the length of the URL is
lesser than a threshold value (49 for the case of this study), it is padded with 0
values. If the URL length is greater than the threshold value, it is padded with
0 values till it’s nearest square higher than the length and then resized down to
the threshold value using bicubic compression.
In the second process, various features are extracted from the URL and
various machine learning are trained using these features in order to find the
model that gives the best results.
After feature extraction, feed the data into the following Machine Learning and Deep
Learning Models and compare their accuracy in detecting malicious URLs:
● Convolutional Neural Network Model(CNN)
● Random Forest Regression Model
● Artificial Neural Network (ANN)
● K- Nearest Neighbour Neural (KNN)
