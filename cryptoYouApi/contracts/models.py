from django.db import models

# Create your models here.

class Contracts(models.Model):
    symbol = models.CharField(max_length=6)
    startdate= models.DateField('start date')
    enddate = models.DateField('end date')
    #epochs = models.IntegerField(  )
    def __str__(self):
        return self.symbol
    def retstartdate(self):
        return self.startdate
    def retenddate(self):
        return self.enddate