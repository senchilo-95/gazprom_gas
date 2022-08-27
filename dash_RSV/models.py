from django.db import models

class gas_futures(models.Model):
    date = models.DateTimeField('DT')
    month_name=models.TextField('MONTH')
    change_price=models.FloatField('CHANGE')
    settle_price=models.FloatField('SETTLE')

    def __str__(self):
        return self.date

    class Meta:
        verbose_name = 'Цена фьючерсов на газ'
        verbose_name_plural = 'Цена фьючерсов на газ'






