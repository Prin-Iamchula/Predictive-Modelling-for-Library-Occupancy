import pandas as pd

# Change the file location before running code.
data_2018 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2018.csv"
data_2019 = "C:/Users/Prin/Desktop/dissertation/Datashare/juno_2019.csv"

df_2018 = pd.read_csv(data_2018, encoding='latin1')
df_2019 = pd.read_csv(data_2019)

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
Relabeling School of user (Category)
'''

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
other_school = [sch for sch in school_list[school_list.TIMESTAMP < 100].index.unique()]+['Override', '\\N']

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
# Extract date from 2018 and 2019
days = []
for i in df['Date'].unique():
    days.append(i)

# Generate corpus
corpus = {}
for day in days:
    data = df[df['Date'] == day]
    corpus[day] = data

# Generate sch_corpus
sch_corpus = {}
for day in days:
    data = df_in[df_in['Date'] == day]
    sch_corpus[day] = data

'''
Export Hourly dataset
'''
def difference(list1):
    time_frame = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13',
                  '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

    diff_time = [value for value in time_frame if value not in list1]
    return diff_time
