from django.shortcuts import render
from django.http import HttpResponse, HttpResponseNotFound
from django.db import models
from django.shortcuts import redirect 
from django.shortcuts import get_object_or_404
import uuid
import openpyxl

# Create your models here.
class Item(models.Model): 
    id = models.UUIDField(
        primary_key = True, default = uuid.uuid4, editable = False
    )
    name = models.CharField(max_length=100, blank = True)

# Create views 
def create_item(request): 
    context = {}

    if request.method == 'POST': 
        name = request.POST['name']

        item = Item(name=name)
        item.save()

        return redirect('item', pk=item.id, permanent=True)
    return render(request,'items/item.html',context)


def item(request, pk): 
    item = get_object_or_404(Item, pk=pk)

# Read Excel file 
def index(request): 
    if "GET" == request.method: 
        return render(request, "postrainer/index.hmtl",{})
    else: 
        excel_file = request.FILES['excel_file']
        wb = openpyxl.load_workbook(excel_file)
        excel_data = list() 
        worksheet = wb["Sheet1"]
        print(worksheet)
        for row in worksheet.iter.rows(): 
            row_data = list() 
            for cell in row: 
                row_data.append(str(cell.value))
            excel_data.append(row_data)
        return render(request, 'postrainer/index.html',{"excel_data": excel_data})
