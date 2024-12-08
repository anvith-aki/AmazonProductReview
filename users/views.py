from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from textblob import TextBlob
from .models import SentimentModel
from .forms import SentimentForm


# Create your views here.
# Registration function
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

# check logid info and see if authorized user or not
def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})

# show home page
def UserHome(request):
    return render(request, 'users/UserHome.html', {})

# to view user dataset
def user_view_dataset(request):
    from django.conf import settings
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path)
    df.drop('reviews.rating', inplace=True, axis=1)
    df.drop('asins', inplace=True, axis=1)
    df.drop('name', inplace=True, axis=1)
    df.drop('reviews.userProvince', inplace=True, axis=1)
    df.drop('reviews.sourceURLs', inplace=True, axis=1)
    df.drop('reviews.doRecommend', inplace=True, axis=1)
    df.drop('reviews.dateSeen', inplace=True, axis=1)
    df.drop('reviews.dateAdded', inplace=True, axis=1)
    df.drop('reviews.date', inplace=True, axis=1)
    df.drop('brand', inplace=True, axis=1)
    df.drop('categories', inplace=True, axis=1)
    df.drop('manufacturer', inplace=True, axis=1)
    df.drop('keys', inplace=True, axis=1)
    df.drop('reviews.didPurchase', inplace=True, axis=1)
    df.drop('reviews.id', inplace=True, axis=1)
    df.drop('reviews.numHelpful', inplace=True, axis=1)
    df.drop('reviews.userCity', inplace=True, axis=1)
    df = df.head(500).to_html
    return render(request, 'users/dataset_view.html', {'data': df})

# Imp func to classify sentiment
def assign_sentiment(reviewText):
    
    blob1 = TextBlob(reviewText)
    if blob1.sentiment.polarity >0 :
        return "Positive"
    elif blob1.sentiment.polarity ==0:
        return "Neutral"
    else:
        return "Negative"

# read dataset and plot graphs
def user_view_sentiment(request):
    from django.conf import settings
    import pandas as pd

    import matplotlib.pyplot as plt

    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path, dtype='unicode')
    df.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer', 'date', 'dateAdded', 'dateSeen',
                  'isPurchase', 'isRecommended', 'reviewsId', 'numHelpful', 'rating', 'sourceURLs', 'reviewText',
                  'reviewTitle', 'city', 'userProvince', 'username']
    df = df.head(500)
    print(df.reviewsId.dtype)
    df.reviewsId.fillna(0.0)
    print(df.head(500))
    df = df.drop('keys', 1)
    df.drop('sourceURLs', 1, inplace=True)
    df.drop(['dateAdded', 'dateSeen'], 1, inplace=True)
    # df.head()
    df.isPurchase.fillna(False, inplace=True)
    df.reviewsId.fillna("", inplace=True)
    df.city.fillna("", inplace=True)
    df.userProvince.fillna("", inplace=True)
    # df.head()
    print(df.describe(include='object'))
    df.dropna(subset=['name'], inplace=True)
    df.describe(include='object')
    df['name'].value_counts()
    sdf = df[['rating', 'reviewText']]
    sdf.head(2)
    sdf['sentiment'] = sdf['reviewText'].apply(assign_sentiment)
    sdf.drop('rating', inplace=True, axis=1)
    sdf = sdf.to_html
    #count reviews
    # x=df['reviewText'].apply(assign_sentiment).value_counts()
    # print(x)
    return render(request, 'users/user_view_sentiment.html', {'data': sdf})


def user_classifiers(request):
    from .utility import amazon_process
    result = amazon_process.start_classification_analysis()


    #---------------------------------------------------------graphs-open

    #piechart    
    from django.conf import settings
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    df = pd.read_csv(path, dtype='unicode')
    df.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer', 'date', 'dateAdded', 'dateSeen',
                  'isPurchase', 'isRecommended', 'reviewsId', 'numHelpful', 'rating', 'sourceURLs', 'reviewText',
                  'reviewTitle', 'city', 'userProvince', 'username']
    df = df.head(500)
    print(df.reviewsId.dtype)
    df.reviewsId.fillna(0.0)
    print(df.head(500))
    df = df.drop('keys', 1)
    df.drop('sourceURLs', 1, inplace=True)
    df.drop(['dateAdded', 'dateSeen'], 1, inplace=True)
    # df.head()
    df.isPurchase.fillna(False, inplace=True)
    df.reviewsId.fillna("", inplace=True)
    df.city.fillna("", inplace=True)
    df.userProvince.fillna("", inplace=True)
    # df.head()
    print(df.describe(include='object'))
    df.dropna(subset=['name'], inplace=True)
    df.describe(include='object')
    df['name'].value_counts()    
    sdf = df[['rating', 'reviewText']]
    sdf.head(2)
    sdf['sentiment'] = sdf['reviewText'].apply(assign_sentiment)
    
    Tasks =sdf['sentiment'].value_counts()

    my_labels = 'Positive','Neutral','Negative'
    # plt.pie(Tasks,labels=my_labels,autopct='%1.1f%%',colors = ['#00994d','#ff751a','#ff4d4d'])
    plt.pie(Tasks,labels=my_labels,autopct='%1.1f%%',explode=[0.1,0,0.1],colors = ['#00994d','#ff751a','#ff4d4d'])
    plt.title('Sentiment')
    plt.axis('equal')
    # plt.savefig("./assets/static/assets/img/piechart.png", bbox_inches="tight", pad_inches=1, transparent=True)
    plt.show(block=True)

        #barGraph
    Reviews = ['Positive','Neutral','Negative']
    InNumber = sdf['sentiment'].value_counts()
    plt.bar(Reviews, InNumber,color = ['#00994d','#ff751a','#ff3333'])
    plt.title('Sentiment of Reviews')
    plt.xlabel('Reviews')
    plt.ylabel('In Number')
    # plt.savefig("./assets/static/assets/img/bar.png", bbox_inches="tight", pad_inches=1, transparent=True)
    plt.show(block=True)

    #--------------------------------------------------graphs-close

    #wordcloud

    # from wordcloud import WordCloud, STOPWORDS
    # text = " ".join(i for i in df.reviewText)
    # stopwords = set(STOPWORDS)
    # wordcloud = WordCloud(stopwords=stopwords, 
    #                     background_color="white").generate(text)
    # plt.figure( figsize=(15,10))
    # plt.imshow(wordcloud, interpolation='bilinear')
    # plt.axis("off")
    # plt.savefig("./assets/static/assets/img/wordcloud.png", bbox_inches="tight", pad_inches=1, transparent=True)
    # # plt.show()

    return render(request, 'users/user_classification_result.html', result)




def app(request):
    form = SentimentForm(request.POST or None)
    context = {}
    if request.method == 'POST':
        if form.is_valid():
            sent = form.cleaned_data.get('Sentence')    # got the sentence
            textAns =assign_sentiment(sent)
            context['text'] = textAns
        else:
            form = SentimentForm()
    
    context['form'] = form
    return render(request, 'users/app.html', context=context)
