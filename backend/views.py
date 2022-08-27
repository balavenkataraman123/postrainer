# Work in progress
from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
import datetime 
import uuid 

def create_item(request): 
    context = {}

    if request.method == 'POST': 
        name = request.POST['name']

        item = Item(name=name)
        item.save()

        return HTTPResponsePermanentRedirect(reverse('item',args=(item,id)))
    return render(request,'items/item.html',context)


def item(request, pk): 
    item = get_object_or_404(Item, pk=pk)
