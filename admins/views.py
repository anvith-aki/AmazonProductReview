from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel


# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


# to view training dataset
def AdminDataset(request):
    from django.conf import settings
    import pandas as pd
    path = settings.MEDIA_ROOT + "\\" + "amazon_reviews.csv"
    # df = pd.read_csv(path)
    df = pd.read_csv(path, skiprows=range(1,101))
    # df.drop('reviews.rating', inplace=True, axis=1)
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
    df = df.head(5000).to_html
    return render(request, 'admins/AdminDataset.html', {'data': df})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')


def admin_classification(request):
    from users.utility import amazon_process
    result = amazon_process.start_classification_analysis()
    return render(request, 'admins/classification_result.html', result)