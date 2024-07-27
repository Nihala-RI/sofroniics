import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from tkinter import *
import string

df=pd.read_csv("E:\datasets for softroniics\spam.csv",encoding='latin1')
df=df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
df.rename(columns={'v1':'labels','v2':'message'},inplace=True)
df.drop_duplicates(inplace=True)
df['labels']=df['labels'].map({'ham':0,'spam':1})
import string


def preprocess_text(message):
    without_punct = [char for char in message if char not in string.punctuation]

    without_punc = "".join(without_punct)

    return [word for word in without_punc.split() if word.lower() not in stopwords.words('english')]

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
df['message'].apply(preprocess_text)

from sklearn.feature_extraction.text import CountVectorizer
x=df['message']
y=df['labels']
cv=CountVectorizer()
x=cv.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB().fit(x_train,y_train)


def sms():
    # creating a list of labels
    classes = ['not spam', 'spam']

    # perform tokenization
    x = cv.transform([e.get()])

    # predict the text
    p = classifier.predict(x)

    # convert the words in string with the help of list
    # s=[str(i) for i in p]
    # a=int("".join(s))

    # show out the final result
    results = ["This message is looking:" + classes[a] for a in p]
    for res in results:
        print(res)

    if [classes[a] for a in p]=='spam':
        classification=Label(root,text=res,font=('helvetica',15,'bold'),fg='red')
        classification.pack()
    else:
        classification=Label(root,text=res,font=('helvetica',15,'bold'),fg='green')
        classification.pack()


root =Tk()
root.title('SpellCheck')
root.geometry('400x400')

head=Label(root,text='SPAM Checker',font=('helvetica',24,'bold'))
head.pack()
e=Entry(root,width=400,borderwidth=5)
e.pack()
b=Button(root,text='Check',font=('helvetica',24,'bold'),fg='white',bg='green',command=sms)
b.pack()
root.mainloop()