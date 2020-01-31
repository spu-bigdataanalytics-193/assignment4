# :two_hearts: Using Spark for Machine Learning :stuck_out_tongue_winking_eye::dog:

![science and tech](https://media.giphy.com/media/8qrrHSsrK9xpknGVNF/giphy.gif)

## Introduction

In this project, we will use spark to do machine learning. We will use `Amazon Product Review Dataset` as our data and analyze reviews using natural language processing (NLP).

Natural Language Processing is the field of computer science to process languages to be able to extract usable information, extract contexts, etc. NLP is used for various ways, some of them goes like in the following list:

- Speech recognition
- Speech Segmentation
- Automatic Summarization
- Word Sense Disambiguation
- Sentiment Analysis
- Question Answering
- Named Entity Recognition (NER)
- Machine Translation
- Part of Speech Tagging (POS)
- Abbreviation Detection

For more information on NLP, [wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing) is a good source for first exposure to the terms and definitions.

## Projects

There will be three groups working on different tasks. If all group members decides to do another NLP task, you are welcome to do so. Following list is the default values for each group. 

Group # | Task Assigned
----- | ----------------
Group 1 | Sentiment Analysis
Group 2 | Topic Modeling
Group 3 | Named Entity Recognition

PS: Groups can **not** do the same task. Sorry.

## Project Information

For our project, there will be three groups working on different NLP tasks. Each group will try to do implement NLP on the dataset, in order to do that, you may have to follow machine learning steps, prepare progress presentations, and finally, present your results at the end of the term.

1. Exploratory Data Analysis (EDA)
2. Data Processing 
3. Implementing Algorithms
4. Validation
5. Testing your Model

PS: If you think there is another step that should be in one of the following lists in below, please inform me, I will update it for future reference.

### 1. EDA

In this part, you will have to achieve **at least 3** of the following list.

1. Prepare graphical representation of `overall`
2. Get the count of most common words
3. Categorize the length of reviews
4. Find lexicality score and perplexity score of the dataset
5. Graph a [dispersion plot](https://www.nltk.org/book/ch01.html) for some words.
6. Find the total # of sentences in the whole dataset
7. Find the total # of words in the whole dataset
9. Find the total # of characters in the whole dataset
10. Find the percentage of special characters (punctuations, numbers, etc) over all words

### 2. Data Processing

Second, and the most important part of the project is to apply text processing to the dataset. 

Again, **at least 3** of the following list should be applied to your processing step. **Bolded** ones **must** be implemented.

1. Removing punctations.
2. **Lowercasing words**
3. Removing stopwords
4. Lemmatization
5. Stemming
6. Removing top n% and lowest n% most used words from dataset
7. Correcting spelling for words
8. Tokenizing review into sentences
9. **Tokenizing review into words**
10. Removing short words (len(word) > n)

### 3. Machine Learning

In this part, we will create our X and y matrices and apply some algorihms to do NLP tasks. In order to do that, **one step** in the following list **must** be implemented.

1. Create [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) matrix.
2. Create [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model) model with Count
3. Create Bag of Words model with Normalized Count
4. Create Doc2Vec representations
3. Create [Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)

Each one of these transformations will create a numerical representation of your words dataset. For example, after applying one of the above transformations, your data should look like in the following image.

![Dictionary Doc Matrix](https://datameetsmedia.com/wp-content/uploads/2017/05/bagofwords.004.jpeg)

As you guessed, this will be your `X`.

Second, you need to prepare your `y` dataset as well. For that purpose, you may use `overall` to derive a ground truth variable for `positiveness`. You may also use other columns available to come up another type of `y`, depending on the NLP task you are trying to achieve.

Third, data should be divided into `training` and `test` splits. This part is essential to test the validity of our machine learning models.

Finally, after you have your `X` and `y`, it is time to apply some algorithms. At least three algorithms must be tried in our project.

Some of these algorithms, however not all, are in the follwing list.

1. KMeans
2. Naive Bayes
3. Logistic Regression
2. SVM, SGDClassifier
3. Birch
4. Latent Dirichlet Allocation (LDA)
5. Hierarchical Dirichlet Process (HDP) Model
6. Latent semantic indexing (LSI) Model
7. Mallet
8. Random Projections (RP) Model
9. TFIDFModel
10. NormModel
11. LogEntropyModel
12. Neural Networks
13. Linear Regression

### 4. Validation

Validation is the part you run some sort of testing code to test your validity of your machine learning model.

It may have (or not) any of the following from below list, you must do **at least one** of the validation.

1. Precision, Recall, and Accuracy
2. Confusion matrix, roc curve, auc
3. Coherence

### 5. Testing 

At the final part, you will use your model to output some results. For example, in sentiment analysis, you may input a review and collect a positiveness score as a result of your model. For topic modeling, you may input a review, get back the topic probability distribution from the model. For Named entity Recognition, from the review you imputted, you may get the entities (special terms, places, etc.) in the review.

You must **at least try one** review that is out of your `training` and `test` dataset. You should print the result of your model. For example, one such example could be following pseudocode.

``` py
test_string = "my review ...." 

with open('my_trained_model.pkl', 'rb') as _file:
    model = _file.read()

results = model.evaluate(test_string)
print(results)

>>> {'positiveness': 0.81123}
```

## Data

This Dataset is an updated version of the Amazon review dataset released in 2014. As in the previous version, this dataset includes reviews (ratings, text, helpfulness votes), product metadata (descriptions, category information, price, brand, and image features), and links (also viewed/also bought graphs). In addition, this version provides the following features

The data is maintained by [Jianmo Ni](https://nijianmo.github.io/) and the [Amazon Product Review Data](https://nijianmo.github.io/amazon/index.html) is available after completing the form.

The data is 34 GB, in total of 233.1 million reviews.

The sample of a review is of the following shape.

``` json
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "vote": 5,
  "style": {
    "Format:": "Hardcover"
  },
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

**For our purpose**, we are only interested in `reviewText`, `summary`, and `overall` parts of the data. We may use these columns for the project.

Column| Usage
------ | --------
reviewText | Will be the used to derive X.
summary | May be used along with reviews.
overall | It will be used as ground truth for machine learning. Will be used to create y.

You may use other part of the dataset as well, such as `unixReviewTime`, for creating dispersion plots.

## Tech Stack and Others

While doing this project, your must use `Apache Spark` in order to do operations. You can also use `pandas` and other packages that are not in `spark` environment as well, but for project purpose, our main focus is to learn pyspark API (Other ways might be computationally expensive, *time consuming*, to run).

PS: Don't confuse *Spark* with *pyspark*. *Spark* is the applcation to do big data analytics, *pyspark* is the python API to interact with Spark. You need both to be installed in the platform you want to use.

You can use standard python or anaconda distribution of python. I suggest you to use standard python, and create virtual environments. 

[Visual Studio Code](https://code.visualstudio.com/) is a nice IDE to use, or you can use other editors as well.

Jupyter has two options, `notebook` and `lab`. They are like IDEs for interactive notebooks.

### How can I run spark?

As you already know from previous classes, we have multiple options.

1. Google Colab
2. Databricks
3. Spark on Docker on your own computer
4. Installing Spark Locally
5. SPU [Data Science Lab](dsl.saintpeters.edu) (DSL) (Thanks to @SaintPeters)
6. [Apache Zeppelin](https://dsl.saintpeters.edu:8443) Notebooks available in SPU DSL.

### How to do the Homework?

Although it is not necessary, I urge you to create a different notebook for each step mentioned in `Project Information`. Keeping all of your code in one notebook is hard for readibility and project tracking. 

Your notebooks should also have some **titles, descriptions, explanations, etc** (Making Cell as Markdown) as well, to make the notebooks self explanatory and visually appealing.

For each section, you **have to present** your results using a PowerPoint presentation. Presentations does not have to be too complicated, as long as they show your key points. **Each week** you should have a progress report of what you have done so far.

As a result, you may have the following folder schema.

``` sh
notebooks \
      |
      --- nb_1_eda.ipynb
      --- nb_2_cleaning.ipynb
      --- nb_3_ml.ipynb
      --- nb_4_validaton.ipynb
      --- nb_5_final.ipynb
presentations \
      |
      --- p_progress_report_1.pptx
      --- p_progress_report_2.pptx
      --- ...
      --- p_final.pptx
```

## What are all these other files?

Following table is will give it a meaning for each file.

File                | Description 
-------             | ----------- 
README.md **        | A descriptive file to give an introduction of current project/ assignment. 
Instructions.md **  | A copy of current README.md file. 
LICENCE **          | The licence of the file that every project should have.
.gitignore          | The file to control which files should be ignored by Git.
*.ipynb             | Assignment notebook with your Spark code. 
