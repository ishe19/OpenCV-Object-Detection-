import os

from flask.helpers import url_for
# import source.flask_app.object_detection
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
# import tensorflow as tf
import object_detection

app = Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()
# file Upload
UPLOAD_FOLDER = os.path.join(path, 'uploads')

# Make directory if uploads is not exists
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['avi', 'mp4', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


# def get_model():
#     model = tf.keras.models.model_from_json(
#         open("/home/rants/PycharmProjects/kbs-project/json_files/object_detection.json", "r").read())
#     model.load_weights('/home/rants/PycharmProjects/kbs-project/models/object_detection.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def upload_form():
        return render_template('upload.html')



# @app.route('/upload')
# def upload_files():
#     analysed = False
#     loading = True
#     upload_dir = "uploads/"
#     if request.method == 'POST':
#         f = request.files['file']
#         f.save("uploads/uploded_video.mp4")
#         # analysed = analysis.newConvert
#         if analysed:
#             loading = False
#             return render_template('index.html', analysis_done = analysed, loading = loading)


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)

        files = request.files.getlist('files[]')

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        flash('File(s) successfully uploaded')
        object_detection.video_to_images("uploads/" + filename)
        object_detection.detect_objects("/home/rants/PycharmProjects/kbs-project/source/flask_app/static/frames")
        return redirect('/')
        # return redirect(url_for('uploaded_file', filename=filename))


@app.route('/test')
def test_route():
    print("Hello")
    return render_template('upload.html')




@app.route('/search', methods = ['POST'])
def search_df():
      query = request.form['search']
      processed_text = query.upper()
      print(query)
      images = object_detection.search_object(item=query)
      print(images)
      return render_template('upload.html', images = images, showImages = True)

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
