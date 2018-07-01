install.packages("RCurl")
install.packages("tokenizers")
install.packages("text2vec")

library(dplyr)
library(RCurl)
library(tokenizers)
library(text2vec)
x <- getURL("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt")
y <- read.delim(text = x)

it1 = itoken(y[1:25000,2] %>% as.character())
it2 = itoken(y[1:25000,2] %>% as.character())

it = itoken(y$document %>% as.character(), progressbar = FALSE)
v = create_vocabulary(it) %>% prune_vocabulary(doc_proportion_max = 0.1, term_count_min = 5)
vectorizer = vocab_vectorizer(v)

dtm1 = create_dtm(it1, vectorizer)
dim(dtm1)

dtm2 = create_dtm(it2, vectorizer)
dim(dtm2)

d1_d2_jac_sim = sim2(dtm1, dtm2, method = "jaccard", norm = "none")

dim(d1_d2_jac_sim)

mat = d1_d2_jac_sim %>% as.matrix()
