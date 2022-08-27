from django.db import models

class gas_supply(models.Model):
    date = models.DateTimeField('date')
    country=models.TextField('country')
    gas_supply=models.FloatField('gas_supply')

    def __str__(self):
        return self.date

    class Meta:
        verbose_name = 'Объемы потребления газа'
        verbose_name_plural = 'Объемы потребления газа'


