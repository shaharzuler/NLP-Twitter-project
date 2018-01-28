def review_to_words( raw_review ):
    '''preproceccing tweets'''
    # Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", raw_review) 
    # Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #  convert the stop words to a set for faster calculations
    stops = set(stopwords.words("english"))                  
    #  Remove stop words
    meaningful_words = [w for w in words if not w in stops]  
    # stemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in meaningful_words]
    # lemmitizer
    wordnet_lemmatizer = WordNetLemmatizer()
    lammatized_words = [wordnet_lemmatizer.lemmatize(word) for word in  stemmed_words  ]
    # Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   