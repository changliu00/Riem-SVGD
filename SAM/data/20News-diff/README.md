# The 20News-different dataset with tf-idf feature

* Created by [Chang Liu][changliu] (<chang-li14@mails.tsinghua.edu.cn> (deprecated); <liuchangsmail@gmail.com>)
  based on the Matlab/Octave version of the [20Newsgroups dataset](http://www.qwone.com/~jason/20Newsgroups/).
* It contains 3 catagories out of the total 20: _rec.sport.baseball_, _sci.space_, _alt.atheism_.
* Dataset for the experiments of the work  
  [Stochastic Gradient Geodesic MCMC Methods](http://papers.nips.cc/paper/6281-stochastic-gradient-geodesic-mcmc-methods)
  ([Chang Liu][changliu], [Jun Zhu][junzhu], [Yang Song][yangsong]. NIPS 2016)

## Main files
* "diff.train.tfidf1.data" with 1,666 documents and "diff.test.tfidf1.data" with 1,107 documents
  are the training and test dataset, respectively.
* Data format:  
  ```
  	[#vocabulary=5000] [#documents= 1666 or 1107]
  	n1 y1 w11:t11 w12:t12 ... w1(n1):t1(n1)
  	n2 y2 w21:t21 w22:t22 ... w2(n2):t2(n2)
  	...
  	[#labels=3]
  ```
  where `n(d)` is the number of different words appearing in document `d`,
  `y(d)` is the label of document `d`,
  `w(d)(i)` is the ID (an integer between 0 and (#vocabulary-1)) of an appearing word in document `d`,
  `t(d)(i)` is the tf-idf value of `w(d)(i)`.

## Other files
* "diff.map" is the correspondences of numbers used by our dataset and the original one that represent a same label.
* "diff.voc" is the shrinked 5,000 vocabulary.
* "diff.idx" is the IDs of chosen words from the original 61,188 vocabulary.

## Details
* The tf-idf feature is converted from the original bag-of-words feature.
* The tf-idf feature of word with ID `v` in document `d` is  
  ```
  	tf-idf(d,v) = tf(d,v) * log( D / (1+df(v)) ),
  ```
  where `tf(d,v)` is the times word `v` appears in document `d`,
  `D` is the number of documents,
  `df(v)` is the document frequency of `v`, i.e. the number of documents containing word `v`.
* The tf-idf feature of document `d` is an `l2`-normalized vector of #vocabulary dimension,
  with value of component `v` proportional to `tf-idf(d,v)`.
* We first collect our training and test data in bag-of-words feature,
  then convert each of both to tf-idf feature by the above procedure.

[changliu]: http://ml.cs.tsinghua.edu.cn/~changliu/index.html
[junzhu]: http://ml.cs.tsinghua.edu.cn/~jun/index.shtml
[yangsong]: https://yang-song.github.io/

