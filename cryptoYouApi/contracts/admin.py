from django.contrib import admin

from .models import Contracts


class ContractsAdmin(admin.ModelAdmin):
    fields = ['pub_date', 'question_text']

admin.site.register(Contracts, ContractsAdmin)