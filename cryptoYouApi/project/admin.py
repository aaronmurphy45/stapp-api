from django.contrib import admin

from .models import pricePredict, Contracts



class pricePredictAdmin(admin.ModelAdmin):
    fields = ['enddate', 'startdate', 'symbol']
    
class ContractsAdmin(admin.ModelAdmin):
    fields = ['enddate', 'startdate', 'symbol']

admin.site.register(pricePredict, pricePredictAdmin)
admin.site.register(Contracts, ContractsAdmin)

