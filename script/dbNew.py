import scipy.io, sqlite3, csv, os
import pandas as pd

conn = sqlite3.connect('../web/landmark.db')
curs = conn.cursor()
conn.commit()

f = open('../data/result_delf_new.csv','r')
csvReader = csv.reader(f)
header = next(csvReader)

for row in csvReader:
    delf_id = (row[0])
    delf_landmarks = (row[1])
    sql = "insert into LMDelf (SID, LID) values (?, ?) "
    curs.execute(sql,(delf_id, delf_landmarks))
    print (delf_id)
    print (delf_landmarks)

df = pd.read_sql_query("select * from LMDelf", conn)
# print(df)


scene_num = pd.read_sql_query("select SUBSTR(SID,-6,3) SID from LMDelf", conn)
df["scene_num"]=scene_num
#print(df)

df[['scene_num']] = df[['scene_num']].apply(pd.to_numeric)
# print(df.dtypes)


csv_path = '../csv'
path, dirs, files = next(os.walk(csv_path))
file_len = len(files)
print('Number of CSVs:', file_len)

for root, dirs, files in os.walk(csv_path):  # loop through startfolders
  for csv in files:
    csv_filename = csv
    print(csv_filename)

df2 = pd.read_csv('../csv/%s' %csv_filename)
# print(df2)

df3 = df2.drop(0)
df3.rename(columns = {df2.columns[0]: "scene_num"}, inplace = True)
df3.rename(columns = {df2.columns[2]: "Timecode"}, inplace=True)
df3[['scene_num']] = df3[['scene_num']].apply(pd.to_numeric)
# print(df3)

result= df[['SID','LID','scene_num']].merge(df3[['Timecode','scene_num']], on='scene_num', how='left')
# print(result)

video_name = pd.read_sql_query("select SUBSTR(SID, 1, INSTR(SID,'-')-1) SID from LMDelf", conn)
VN = video_name +'.mp4'
result["VN"]=VN
print(result)

result.to_sql('LMTime', conn, index=False, if_exists='append')


scenes_path = '../scenes'
# copy scenes
for root, dirs, files in os.walk(scenes_path):  # loop through startfolders
  for pic in files:
    filename = pic

    for sid in result["SID"]:
      if filename == sid+".jpg":
        print("copy " + filename + " to static folder")
        cmd = "cp ../scenes/%s ../web/static" % (filename)
        os.system(cmd)

    # copy thumbnail image
    if filename.split('-')[2] == "001" and filename.split('-')[3] == "01.jpg":
        print("copy thumbnail " + filename + " to static folder")
        cmd2 = "cp ../scenes/%s ../web/static" % (filename)
        os.system(cmd2)

# reset
# delete all rows in a table
curs.execute("DELETE FROM LMDelf")

# delete files
os.chdir("../data")
os.system("pwd")
os.system("rm prediction_new.csv")
os.system("rm result_delf_new.csv")

os.chdir("../csv")
os.system("pwd")
os.system("rm *")

os.chdir("../scenes")
os.system("pwd")
os.system("rm -f *")


# close db
conn.commit()

f.close()
conn.close()
