import pandas as pd
import collections
from zipfile import ZipFile
import warnings
warnings.filterwarnings("ignore")

print('Loading data...')
print('')
# Change the file location before running the code below.
data_2018 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2018.csv"
data_2019 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2019.csv"

# Change the file location before running the code below.
weather2018_zip = "C:/Users/Prin/Desktop/dissertation/Datashare/weather2.zip"
weather2019_zip = "C:/Users/Prin/Desktop/dissertation/Datashare/weather_2019(2).zip"
daily_weather_zip = "C:/Users/Prin/Desktop/dissertation/Datashare/daily_weather.zip"

# Please change and specify the exported location here!!!!
export_loc = 'C:/Users/Prin/Desktop/dissertation/Datashare'

df_2018 = pd.read_csv(data_2018, encoding='latin1')
df_2019 = pd.read_csv(data_2019)

print('Cleaning data...')
print('')
# Drop the error rows
df_2018 = df_2018.drop(df_2018.index[1112133:1112150])

# Concatenate 2018 and 2019 dataset
df = pd.concat([df_2018, df_2019], ignore_index=True)

# Split Date and Time
df[['Date', 'Time']] = df.TIMESTAMP.str.split(" ", expand=True)

# Add O'clcok column
df['Oclock'] = df.Time.str.split(":", expand=True)[0]

# Generate numeric value for IN and OUT direction.
inn = []
out = []
for d in df['Direction']:
    # For IN column, count 1 if direction is 'I' and count 0 if direction is 'O'.
    if d == 'I':
        inn.append(1)
        out.append(0)
    # For OUT column, count 0 if direction is 'I' and count 1 if direction is 'O'.
    elif d == 'O':
        inn.append(0)
        out.append(1)

# Add IN and OUT column to dataframe.
df['IN'] = inn
df['OUT'] = out

'''
Relabeling "Category1"
'''

print('Relabeling Category1...')
print('')
# Relabel 'Postgraduate'
df = df.replace({'Category1': {'Postgraduate Taught': 'Postgraduate', 'Postgraduate Research': 'Postgraduate',
                               'PG': 'Postgraduate'}})

# Relabel 'Undergraduate'
df = df.replace({'Category1': {'Under Graduate': 'Undergraduate', 'undergraduate': 'Undergraduate'}})

# Relabel 'Assistive-Tech'
df = df.replace({'Category1': {'Assistive Tech Enabled': 'Assistive-Tech', 'Assistive Tech Enabled ': 'Assistive-Tech',
                               'Assistive Tech Enabled   ': 'Assistive-Tech',
                               'Assistive Tech Enabled  ': 'Assistive-Tech', 'Assistive tech Enabled': 'Assistive-Tech',
                               'Assistive Tech ': 'Assistive-Tech', 'Assisitve Tech Enabled ': 'Assistive-Tech',
                               'assistive Tech Enabled': 'Assistive-Tech', 'Assistive Tech': 'Assistive-Tech'}})

# Relabel 'Library-Staff'
df = df.replace({'Category1': {'Library Staff': 'Library-Staff'}})

# Relabel 'Staff'
df = df.replace({'Category1': {'Research': 'Staff', 'Associate': 'Staff', 'ASSOCIATE': 'Staff'}})

# Relabel 'Sconul'
df = df.replace({'Category1': {'Sconul Band B': 'Sconul', 'Sconul Band A': 'Sconul', 'Sconul Band C': 'Sconul',
                               'Sconul Reference Only': 'Sconul'}})

# Relabel 'Default'
df = df.replace({'Category1': {'default': 'Default', 'Defaut': 'Default', 'DEFAULT': 'Default',
                               'Reference': 'Default'}})

# Relabel 'BN'
df = df.replace({'Category1': {'BN01': 'BN', 'BN07': 'BN'}})

'''
Identify the category of each user when enters to library for every single row.
By counting 1 for the category which belongs to user and 0 for other category.
'''

# Generate list for each category
postgrad = []
ungrad = []
assist_tech = []
library_staff = []
staff = []
sconul = []
default = []
BN = []
other = []

addition = ['Postgraduate', 'Undergraduate', 'Assistive-Tech', 'Library-Staff', 'Staff', 'Sconul', 'Default', 'BN']

# Append 1 for the category which user are.
for d, c in zip(df['Direction'], df['Category1']):
    if d == 'I' and c == 'Postgraduate':
        postgrad.append(1)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Undergraduate':
        postgrad.append(0)
        ungrad.append(1)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Assistive-Tech':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(1)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Library-Staff':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(1)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Staff':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(1)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Sconul':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(1)
        default.append(0)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'Default':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(1)
        BN.append(0)
        other.append(0)

    elif d == 'I' and c == 'BN':
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(1)
        other.append(0)

    elif d == 'I' and c not in addition:
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(1)

    else:
        postgrad.append(0)
        ungrad.append(0)
        assist_tech.append(0)
        library_staff.append(0)
        staff.append(0)
        sconul.append(0)
        default.append(0)
        BN.append(0)
        other.append(0)

# Add columns of each category to dataframe
df['Postgraduate'] = postgrad
df['Undergraduate'] = ungrad
df['Assistive-Tech'] = assist_tech
df['Library-Staff'] = library_staff
df['Staff'] = staff
df['Sconul'] = sconul
df['Default'] = default
df['BN'] = BN
df['Other'] = other

'''
Relabeling School of user (Category3)
'''

print('Relabeling School(Category3)...')
print('')
# Relabel 'School of Business and Management'
df = df.replace({'Category3': {'School of Business, Management': 'School of Business and Management',
                               'School of Business, Managemen': 'School of Business and Management',
                               'University of Sussex Business ': 'School of Business and Management'}})

# Relabel 'IDS'
df = df.replace({'Category3': {'Ids': 'IDS'}})

# Relabel 'School of Education and Social'
df = df.replace({'Category3': {'School of Education': 'School of Education and Social'}})

# Relabel 'School of Engineering and Informative'
df = df.replace({'Category3': {'School of Engineering and Info': 'School of Engineering and Informative'}})

# Relabel 'School of English'
df = df.replace({'Category3': {'English': 'School of English'}})

# Relabel 'School of Mathematical and Physics'
df = df.replace({'Category3': {'School of Mathematical and Phy': 'School of Mathematical and Physics',
                               'Physics': 'School of Mathematical and Physics'}})

# Relabel 'School of History and Art History'
df = df.replace({'Category3': {'School of History, Art History': 'School of History and Art History'}})

# Relabel 'School of Media, Film and Music'
df = df.replace({'Category3': {'School of Media, Film and Musi': 'School of Media, Film and Music',
                               'School of Media': 'School of Media, Film and Music'}})

# Relabel 'School of Law, Politics and Social'
df = df.replace({'Category3': {'School of Law, Politics and So': 'School of Law, Politics and Social'}})

# Relabel 'Assistive-Tech'
df = df.replace({'Category3': {'Brighton and Sussex Medical Sc': 'Brighton and Sussex Medical School',
                               'Brighton & Sussex Medical Scho': 'Brighton and Sussex Medical School',
                               'Brighton & Sussex Medical Sch': 'Brighton and Sussex Medical School',
                               'Brighton & Sussex medical Scho': 'Brighton and Sussex Medical School',
                               'Bsms': 'Brighton and Sussex Medical School',
                               'BSMS': 'Brighton and Sussex Medical School'}})

# Relabel 'School of Environment and Technology'
df = df.replace({'Category3': {'School of Environment and Tech': 'School of Environment and Technology'}})

# Relabel 'School of Sport and Service Management'
df = df.replace({'Category3': {'School of Sport and Service Ma': 'School of Sport and Service Management'}})

# Relabel 'School of Applied Social Science'
df = df.replace({'Category3': {'School of Applied Social Scien': 'School of Applied Social Science'}})

# Relabel 'University of Brighton'
df = df.replace({'Category3': {'University of Brighton Interna': 'University of Brighton',
                               'UoB': 'University of Brighton'}})

# Relabel 'School of Architecture & Design'
df = df.replace({'Category3': {'School of Architecture & Desig': 'School of Architecture & Design'}})

# Relabel 'Institute of Development Studies'
df = df.replace({'Category3': {'Institute of Development Studi': 'Institute of Development Studies'}})

# Relabel 'Library'
df = df.replace({'Category3': {'LIBRARY': 'Library'}})

# Relabel 'Centre for Learning and Teaching'
df = df.replace({'Category3': {'Centre for Learning and Teachi': 'Centre for Learning and Teaching'}})

'''
Generate new dataframe to identify school of each user.
'''
# New dataframe contain only "IN"
df_in = df[df.Direction == 'I']

# Check number of user for each school
school_list = df_in[['TIMESTAMP', 'Category3']].groupby('Category3').count()
school_list.sort_values('TIMESTAMP')

# Store school name which has number of user over than 100
school = [sch for sch in school_list[school_list.TIMESTAMP > 100].index.unique()]

# Remove Unknown school name
school.remove('Override')
school.remove('\\N')

# Filtering out the school which lower than 100 user, and append to other_school list
other_school = [sch for sch in school_list[school_list.TIMESTAMP < 100].index.unique()] + ['Override', '\\N']

# Create all school columns to dataframe
for sc in school:
    df_in[sc] = 0

df_in['other_school'] = 0

# Identifying school for each user
for col in school:
    df_in.loc[df_in['Category3'] == col, [col]] = 1

for col in other_school:
    df_in.loc[df_in['Category3'] == col, ['other_school']] = 1

df_in.loc[df_in['Category3'].isna(), ['other_school']] = 1

'''
Store the every day raw data from 2018 to 2019 in "corpus",
and "sch_corpus" for school dataframe
'''

print('Generating corpus...')
print('')
# Extract date from 2018 and 2019
days = []
for i in df['Date'].unique():
    days.append(i)

# Generate corpus
corpus = {}
for day in days:
    data = df[df['Date'] == day]
    corpus[day] = data

print('Generating sch_copus...')
print('')
# Generate sch_corpus
sch_corpus = {}
for day in days:
    data = df_in[df_in['Date'] == day]
    sch_corpus[day] = data

'''
Making the term date adding function
'''

def extract_period(s, u, month, year):
    alist = []
    for i in range(s, u + 1):
        i = str(i)
        if len(i) == 1:
            i = '0' + i
        alist.append(year + '-' + month + '-' + i)
    return alist


# Christmas vacation period 1
christmas_vac1 = extract_period(1, 7, '01', '2018') + extract_period(1, 6, '01', '2019')

# Christmas vacation period 2
christmas_vac2 = extract_period(15, 31, '12', '2018') + extract_period(14, 31, '12', '2019')

# Private study period1
private_study1 = extract_period(8, 10, '01', '2018') + extract_period(7, 9, '01', '2019')

# Private study period2
private_study2 = extract_period(14, 16, '05', '2018') + extract_period(13, 15, '05', '2019')

# Mid-year assessment period
mid_assess = extract_period(11, 23, '01', '2018')
mid_assess = mid_assess + extract_period(10, 22, '01', '2019')

# Winter graduation
win_grad = extract_period(24, 26, '01', '2018')
win_grad = win_grad + extract_period(23, 25, '01', '2019')

# Inter-session week
inter_ses = extract_period(27, 31, '01', '2018') + extract_period(1, 4, '02', '2018')
inter_ses = inter_ses + extract_period(26, 31, '01', '2019') + extract_period(1, 3, '02', '2019')

# Easter teaching break and spring vacation
easter_break = extract_period(24, 31, '03', '2018') + extract_period(1, 8, '04', '2018') + extract_period(13, 28, '04',
                                                                                                          '2019')

# year-end assessment period (summer term)
year_assess = extract_period(17, 31, '05', '2018') + extract_period(1, 15, '06', '2018')
year_assess = year_assess + extract_period(16, 31, '05', '2019') + extract_period(1, 14, '06', '2019')

# Summer graduation
summer_grad = extract_period(23, 27, '07', '2018') + extract_period(22, 26, '07', '2019')

# Summer vacation assessment period
summer_assess = extract_period(20, 31, '08', '2018') + extract_period(1, 7, '09', '2018')
summer_assess = summer_assess + extract_period(19, 31, '08', '2019') + extract_period(1, 6, '09', '2019')

# Arrival Weekend
arrival_weekend = extract_period(15, 17, '09', '2018') + extract_period(21, 22, '09', '2019')

# Autumn Term
autumn_term = extract_period(18, 30, '09', '2018') + extract_period(1, 31, '10', '2018')
autumn_term = autumn_term + extract_period(1, 30, '11', '2018') + extract_period(1, 14, '12', '2018')

autumn_term = autumn_term + extract_period(23, 30, '09', '2019') + extract_period(1, 31, '10', '2019')
autumn_term = autumn_term + extract_period(1, 30, '11', '2019') + extract_period(1, 13, '12', '2019')

# Spring term
spring_term = extract_period(5, 28, '02', '2018') + extract_period(1, 31, '03', '2018')
spring_term = spring_term + extract_period(1, 30, '04', '2018') + extract_period(1, 13, '05', '2018')

spring_term = spring_term + extract_period(4, 28, '02', '2019') + extract_period(1, 31, '03', '2019')
spring_term = spring_term + extract_period(1, 30, '04', '2019') + extract_period(1, 12, '05', '2019')

# Summer vacation
summer_vac = extract_period(16, 30, '06', '2018') + extract_period(1, 31, '07', '2018')
summer_vac = summer_vac + extract_period(1, 31, '08', '2018') + extract_period(1, 15, '09', '2018')

summer_vac = summer_vac + extract_period(15, 30, '06', '2019') + extract_period(1, 31, '07', '2019')
summer_vac = summer_vac + extract_period(1, 31, '08', '2019') + extract_period(1, 21, '09', '2019')


def add_term(df):
    df['Term_dates'] = 0
    for day in df.Date:
        if day in christmas_vac1:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Christmas vacation 1'

        elif day in christmas_vac2:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Christmas vacation 2'

        elif day in private_study1:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Private study 1'

        elif day in private_study2:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Private study 2'

        elif day in mid_assess:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Mid-year assessment'

        elif day in win_grad:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Winter graduation'

        elif day in inter_ses:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Inter-session week'

        elif day in easter_break:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Easter teaching break'

        elif day in year_assess:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Year-end assessment'

        elif day in summer_grad:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Summer graduation'

        elif day in summer_assess:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Summer vacation assessment'

        elif day in arrival_weekend:
            df.loc[df['Date'] == day, ['Term_dates']] = 'Arrival Weekend'

    for i, day in zip(df.Term_dates, df.Date):
        if i == 0:
            if day in autumn_term:
                df.loc[df['Date'] == day, ['Term_dates']] = 'Autumn term'

            elif day in spring_term:
                df.loc[df['Date'] == day, ['Term_dates']] = 'Spring term'

            elif day in summer_vac:
                df.loc[df['Date'] == day, ['Term_dates']] = 'Summer vacation'
        else:
            pass

    return df


'''
Generating Hourly dataframe
'''

def difference(list1):
    time_frame = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
                  '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    diff_time = [value for value in time_frame if value not in list1]
    return diff_time

def fill_diference(dataframe, data='hourly'):
    alist = []
    for i in dataframe['Oclock']:
        alist.append(i)
    diff_time = difference(alist)

    if data == 'hourly':
        for time in diff_time:
            dataframe = dataframe.append(
                {'Oclock': time, 'IN': 0, 'OUT': 0, 'Postgraduate': 0, 'Undergraduate': 0, 'Assistive-Tech': 0,
                 'Library-Staff': 0, 'Staff': 0, 'Sconul': 0, 'Default': 0, 'BN': 0, 'Other': 0}, ignore_index=True)

    if data == 'school':
        for time in diff_time:
            dataframe = dataframe.append({'Oclock': time}, ignore_index=True)
            dataframe = dataframe.fillna(0)

    if data == 'weather':
        for time in diff_time:
            dataframe = dataframe.append({'Oclock': time, 'Time': time + ':20'}, ignore_index=True)
        dataframe['Date'] = dataframe.Date.fillna(method='ffill')  # fill Date
        dataframe['Date'] = dataframe.Date.fillna(method='bfill')

    dataframe = dataframe.sort_values(by='Oclock')
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


hourly_corpus = {}
print('Generating hourly dataframe...')
for day in days:
    date_data = corpus[day]
    hourly_df = date_data.groupby('Oclock')['IN', 'OUT', 'Postgraduate', 'Undergraduate', 'Assistive-Tech',
                                            'Library-Staff', 'Staff', 'Sconul', 'Default', 'BN', 'Other'].sum()
    hourly_df = hourly_df.reset_index()
    hourly_df = fill_diference(hourly_df)
    hourly_df["Date"] = day
    cols = hourly_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    hourly_df = hourly_df[cols]

    hourly_corpus[day] = hourly_df

hourly_data = pd.concat(hourly_corpus, ignore_index=True)
print('Adding term dates to hourly dataframe...')
hourly_data = add_term(hourly_data)

# Create Hourly index
hour = []
for h in hourly_data.Oclock:
    h = str(h)
    if len(h) == 1:
        hour.append('0'+h)
    else:
        hour.append(h)

hourly_data['Hour'] = hour
hourly_data['Hour'] = hourly_data.Date+" "+hourly_data.Hour+':'+'00'+':'+'00'
hourly_data = hourly_data.set_index('Hour')
print('Exporting hourly_data.csv to {}...'.format(export_loc))
hourly_data.to_csv(export_loc+'/hourly_data.csv')
print('Done!')
print('')

'''
Generating Hourly + school dataframe
'''
sch_col = ['IN'] + school + ['other_school']
sch_hourly_corpus = {}
print('Generating hourly+school dataframe...')
for day in days:
    date_data = sch_corpus[day]
    sch_hourly_df = date_data.groupby('Oclock')[sch_col].sum()
    sch_hourly_df = sch_hourly_df.reset_index()
    sch_hourly_df = fill_diference(sch_hourly_df, 'school')
    sch_hourly_df["Date"] = day
    cols = sch_hourly_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    sch_hourly_df = sch_hourly_df[cols]

    sch_hourly_corpus[day] = sch_hourly_df

sch_hourly_data = pd.concat(sch_hourly_corpus, ignore_index=True)
print('Adding term dates to hourly+school dataframe...')
sch_hourly_data = add_term(sch_hourly_data)

# Adding hour index
sch_hourly_data['Hour'] = hour
sch_hourly_data['Hour'] = sch_hourly_data.Date+" "+sch_hourly_data.Hour+':'+'00'+':'+'00'
sch_hourly_data = sch_hourly_data.set_index('Hour')
print('Exporting sch_hourly_data.csv to {}...'.format(export_loc))
sch_hourly_data.to_csv(export_loc+'/sch_hourly_data.csv')
print('Done!')
print('')

'''
Generating Daily dataframe
'''
print('Generating daily dataframe...')
daily_data = df.groupby('Date')['IN', 'OUT', 'Postgraduate', 'Undergraduate', 'Assistive-Tech', 'Library-Staff',
                                'Staff', 'Sconul', 'Default', 'BN', 'Other'].sum()

print('Adding term dates to daily dataframe...')
daily_data = daily_data.reset_index()
daily_data = add_term(daily_data)
print('Exporting daily_data.csv to {}...'.format(export_loc))
daily_data.to_csv(export_loc+'/daily_data.csv')
print('Done!')
print('')

print('Generating sch_daily dataframe...')
sch_daily_data = sch_hourly_data.groupby('Date')[sch_col].sum()
print('Adding term dates to sch_daily dataframe...')
sch_daily_data = sch_daily_data.reset_index()
sch_daily_data = add_term(sch_daily_data)
print('Exporting sch_daily_data.csv to {}...'.format(export_loc))
sch_daily_data.to_csv(export_loc+'/sch_daily_data.csv')
print('Done!')
print('')

'''
Applying weather condition data to generate weather dataframe.
'''

# This function use to round the time to hourly scale
def remove50(df):
    df['minutes'] = df.Time.str.split(":", expand=True)[1]
    df = df[df.minutes != '50 AM']
    df = df[df.minutes != '50 PM']
    df = df.drop(columns='minutes')
    return df

# This function use to convert the time to 24 hour unit
def convert_time(df):
    result = []
    for time in df.Time:
        hour, mins = time.split(':')

        if len(hour) == 1:
            hour = '0' + hour

        if mins[-2:] == 'AM' and hour == '12':
            hour = '00'
            result.append(hour + ':' + mins[:2])

        elif mins[-2:] == 'AM':
            result.append(hour + ':' + mins[:2])

        elif mins[-2:] == 'PM' and hour == '12':
            result.append(hour + ':' + mins[:2])

        else:
            hour = str(int(hour) + 12)
            result.append(hour + ':' + mins[:2])

    df.Time = result
    return df

# This function use to convert temperature from F to C
def convert_temp(df):
    new_tem = []
    new_d_point = []

    for tem, d_point in zip(df['Temperature(F)'], df['Dew_point(F)']):
        new_tem.append(round((tem - 32) * 5 / 9))
        new_d_point.append(round((d_point - 32) * 5 / 9))

    df['Temperature(F)'] = new_tem
    df['Dew_point(F)'] = new_d_point
    df = df.rename(columns={'Temperature(F)': 'Temperature(C)', 'Dew_point(F)': 'Dew_point(C)',
                            'Wind_Guest(mph)': 'Wind_Gust(mph)'})
    return df

# Import and transform weather condition data
def weather_zip_to_corpus(filezip):
    # retrieve file name in archive
    with ZipFile(filezip, 'r') as myzip:
        files_list = myzip.namelist()
        files_list = files_list[1:]

    # Create corpus (dict)
    weather_corpus = {}
    for file in files_list:

        wet, year, month, day = file.split('-')
        day, csv = day.split('.')
        wet1, wet2 = wet.split('/')
        if len(day) == 1:
            day = '0' + day
        if len(month) == 1:
            month = '0' + month

        date = year + '-' + month + '-' + day
        # rename key for weather corpus base on file name
        file_name = wet2 + '_' + year + '_' + month + '_' + day

        weather_zip = ZipFile(filezip)
        df_temp = pd.read_csv(weather_zip.open(file)).dropna()

        df_temp = remove50(df_temp)
        df_temp = convert_time(df_temp)
        df_temp = convert_temp(df_temp)

        # Add date of data to df
        df_temp['Date'] = date

        # Add Oclock column and fill Nan
        df_temp['Oclock'] = df_temp.Time.str.split(":", expand=True)[0]
        df_temp = fill_diference(df_temp, data='weather')

        cols = df_temp.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_temp = df_temp[cols]

        weather_corpus[file_name] = df_temp
    return weather_corpus

print('Generating weather corpus...')
weather2018_corpus = weather_zip_to_corpus(weather2018_zip)
weather2019_corpus = weather_zip_to_corpus(weather2019_zip)

weather2018_corpus = collections.OrderedDict(sorted(weather2018_corpus.items()))
weather2019_corpus = collections.OrderedDict(sorted(weather2019_corpus.items()))
allweather_corpus = {**weather2018_corpus, **weather2019_corpus}

print('Generating weather dataframe...')
allweather_data = pd.concat(allweather_corpus, ignore_index=True)
print('')

print('Merging hourly data with weather data...')
wet_hour_df = pd.merge(hourly_data, allweather_data, on=['Date', 'Oclock'], how='outer')
wet_hour_df = wet_hour_df.drop(7176)
wet_hour_df = wet_hour_df.drop(15865)
wet_hour_df = wet_hour_df[wet_hour_df['Date'] != '2018-01-01']
wet_hour_df = wet_hour_df[wet_hour_df['Date'] != '2018-12-25']
wet_hour_df = wet_hour_df[wet_hour_df['Date'] != '2018-12-26']
wet_hour_df = wet_hour_df.reset_index().drop(columns='index')
print('Exporting wet_hour_df.csv to {}...'.format(export_loc))
wet_hour_df.to_csv(export_loc+'/wet_hour_df.csv')
print('Done!')
print('')

'''
Generating weather daily data
'''

def daily_weather_zip_to_corpus(filezip):
    # retrieve file name in archive
    with ZipFile(filezip, 'r') as myzip:
        files_list = myzip.namelist()
        files_list = files_list[1:]

    # Create corpus (dict)
    weather_corpus = {}
    for file in files_list:

        file_name = str(file)[14:-4]

        weather_zip = ZipFile(filezip)
        df_temp = pd.read_csv(weather_zip.open(file)).dropna()

        hi_tem = []
        low_tem = []

        for hi, low in zip(df_temp['temp_hi'], df_temp['temp_low']):
            hi = int(hi[:2])
            low = int(low[:2])
            hi_tem.append(round((hi - 32) * 5 / 9))
            low_tem.append(round((low - 32) * 5 / 9))

        df_temp['temp_hi'] = hi_tem
        df_temp['temp_low'] = low_tem

        # Add month of data to df
        df_temp['Month'] = file_name[14:]

        cols = df_temp.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        df_temp = df_temp[cols]

        weather_corpus[file_name] = df_temp
    return weather_corpus

daily_weather_corpus = daily_weather_zip_to_corpus(daily_weather_zip)

def con_day(day):
    if len(str(day)) == 1:
        return '0{}'.format(day)
    else:
        return str(day)

print('Generating daily weather data...')
daily_weather_data = pd.concat(daily_weather_corpus, ignore_index = True)
daily_weather_data['day'] = [con_day(i) for i in daily_weather_data.Day]
daily_weather_data['Date'] = daily_weather_data.Month.str.split('_', expand=True)[0]+'-'\
                             + daily_weather_data.Month.str.split('_', expand=True)[1]+'-'+daily_weather_data.day
daily = daily_data.copy()
daily = daily.reset_index()
print('Merging daily data with weather data...')
daily_x_wet = pd.merge(daily, daily_weather_data, on=['Date'], how='outer')
print('Exporting daily_x_wet.csv to {}...'.format(export_loc))
daily_x_wet.to_csv(export_loc+'/daily_x_wet.csv')
print('Done!')
print('')
