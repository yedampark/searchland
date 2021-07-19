from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask_uploads import UploadSet, configure_uploads, ALL	# UploadSet: upload object
import sqlite3
import os

app = Flask(__name__)

videos = UploadSet('videos', ALL)

app.config['DEBUG'] = True
app.config['UPLOADED_VIDEOS_DEST'] = 'static'
configure_uploads(app, videos)

# connect SQLite DB
conn = sqlite3.connect("landmark.db", check_same_thread=False)
cur = conn.cursor()

@app.route('/', methods=['GET', 'POST'])
def index():
    posts = []

    # search
    if request.args.get('submit') == 'search':
        query = request.args.get('query')

        sql = "SELECT DISTINCT VID, V.VN FROM LMTime T, LMInfo I, VInfo V WHERE T.LID = I.LID AND T.VN = V.VN AND I.LN LIKE '%" + query +"%'"
        cur.execute(sql)
        posts = cur.fetchall()

        if query == "":
            posts = [];

    # view
    if request.args.get('view'):
        id = request.args.get('view')
        search_word = request.args.get('search_word')

        print("view "+search_word)
        return redirect('view/'+str(id)+'/'+search_word)

    # upload
    if request.args.get('submit') == 'upload':
        return redirect('upload_video')

    # nav_search
    if request.args.get('submit') == 'nav_search':
        query = request.args.get('nav_query')
        return redirect('/?query='+query+'&submit=search')

    return render_template('index.html', posts=posts)


@app.route('/view/<int:id>/<string:search_word>', methods=['GET', 'POST'])
def view(id, search_word):
    results = []

    sql = "SELECT VID, V.VN, SID, Timecode FROM LMTime T, LMInfo I, VInfo V WHERE T.LID = I.LID AND T.VN = V.VN AND I.LN LIKE '%" + search_word + "%'" + "AND V.VID= " + str(id)
    cur.execute(sql)
    results = cur.fetchall() # list
    results.sort()

    for r in results:
        print(r)

    # nav_search
    if request.args.get('submit') == 'nav_search':
        query = request.args.get('nav_query')
        return redirect('/?query='+query+'&submit=search')

    return render_template('view.html', results=results)


@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        filesize = request.cookies.get("filesize")
        file = request.files["file"]
        filename = videos.save(file) # save videos to directory

        res = make_response(jsonify({"message": f"{file.filename} uploaded"}), 200)

        os.system('scenedetect --input ./static/%s --output ../scenes detect-content list-scenes save-images --num-images 1 ' %filename)
        print('pyscenedetect done')

        os.system('python ../script/predictNew.py')
        print('predict done')

        os.system('python ../script/delfNew.py')
        print('delf done')

        os.system('python ../script/dbNew.py')
        #insert data into VInfo
        cur.execute("INSERT INTO VInfo(VN) VALUES (:VN)",{'VN': file.filename})
        print('db done')

        conn.commit()

        return res
        
    # nav_search
    if request.args.get('submit') == 'nav_search':
        query = request.args.get('nav_query')
        return redirect('/?query='+query+'&submit=search')        

    return render_template('upload_video.html')

if __name__ == '__main__':
    app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0', port=5000)
