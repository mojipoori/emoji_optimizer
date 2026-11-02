# وارد کردن کتابخانه‌های مورد نیاز
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# تنظیم تم و استایل seaborn
sns.set_theme(style="white",  # استایل سفید برای پس‌زمینه
              rc={"axes.facecolor": (0, 0, 0, 0)})  # پس‌زمینه شفاف برای محورها

# ایجاد داده‌های تصادفی
rs = np.random.RandomState(1979)  # تنظیم seed=1979 برای تولید اعداد تصادفی یکسان
x = rs.randn(500)  # تولید 500 عدد تصادفی با توزیع نرمال (میانگین=0, انحراف معیار=1)
g = np.tile(list("ABCDEFGHIJ"), 50)  # ایجاد 10 گروه (A تا J) که هر کدام 50 بار تکرار می‌شوند
df = pd.DataFrame(dict(x=x, g=g))  # ایجاد DataFrame با ستون‌های x (داده‌ها) و g (گروه‌ها)
m = df.g.map(ord)  # تبدیل حروف به اعداد ASCII (A=65, B=66, ...)
df["x"] += m  # اضافه کردن offset به هر گروه (هر گروه در موقعیت متفاوتی قرار می‌گیرد)

# ایجاد شبکه نمودار با FacetGrid
pal = sns.cubehelix_palette(10,  # تعداد رنگ‌ها = 10
                           rot=-.25,  # چرخش پالت = -0.25
                           light=.7)  # روشنایی = 0.7
g = sns.FacetGrid(df,  # DataFrame داده‌ها
                  row="g",  # هر گروه در یک ردیف جداگانه
                  hue="g",  # رنگ‌بندی بر اساس گروه
                  aspect=15,  # نسبت عرض به ارتفاع = 15
                  height=.5,  # ارتفاع هر زیرنمودار = 0.5
                  palette=pal)  # استفاده از پالت رنگی تعریف شده

# رسم نمودارهای چگالی (KDE) در دو مرحله
g.map(sns.kdeplot, "x",  # رسم KDE برای ستون x
      bw_adjust=.5,  # تنظیم پهنای باند = 0.5 (نرم‌تر کردن منحنی)
      clip_on=False,  # عدم محدود کردن نمودار به محورها
      fill=True,  # پر کردن زیر منحنی
      alpha=1,  # شفافیت = 1 (کاملاً مات)
      linewidth=1.5)  # ضخامت خط = 1.5

g.map(sns.kdeplot, "x",  # رسم مجدد KDE برای خط دور
      clip_on=False,  # عدم محدود کردن نمودار به محورها
      color="w",  # رنگ سفید
      lw=2,  # ضخامت خط = 2
      bw_adjust=.5)  # تنظیم پهنای باند = 0.5

# اضافه کردن خط مرجع در y=0
g.refline(y=0,  # موقعیت خط در y=0
          linewidth=2,  # ضخامت خط = 2
          linestyle="-",  # نوع خط = خط صاف
          color=None,  # استفاده از رنگ‌بندی گروه‌ها
          clip_on=False)  # عدم محدود کردن خط به محورها

# تعریف تابع برای اضافه کردن برچسب به هر نمودار
def label(x, color, label):  # x=داده‌ها, color=رنگ گروه, label=نام گروه
    ax = plt.gca()  # دریافت محور فعلی
    ax.text(0, .2,  # موقعیت x=0, y=0.2
            label,  # متن برچسب
            fontweight="bold",  # وزن فونت = پررنگ
            color=color,  # رنگ متن = رنگ گروه
            ha="left",  # تراز افقی = چپ
            va="center",  # تراز عمودی = وسط
            transform=ax.transAxes)  # استفاده از مختصات نسبی محور

# اعمال تابع برچسب‌گذاری
g.map(label, "x")  # اعمال تابع label به ستون x

# تنظیم فاصله بین زیرنمودارها برای ایجاد همپوشانی
g.figure.subplots_adjust(hspace=-.25)  # فاصله عمودی منفی = همپوشانی

# حذف جزئیات محورها که با همپوشانی سازگار نیستند
g.set_titles("")  # حذف عنوان‌ها (خالی)
g.set(yticks=[],  # حذف تیک‌های محور y
      ylabel="")  # حذف برچسب محور y
g.despine(bottom=True,  # حذف خط پایین
          left=True)  # حذف خط چپ

# نمایش نمودار
plt.show()