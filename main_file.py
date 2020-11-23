import os
import cv2 as cv
import numpy as np
import base64
import string
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request

devnagarik_word=['ञ','ट','ठ','ड','ढ','ण','त','थ','द','ध','क','न','प','फ','ब','भ','म','य',
                 'र','ल','व','ख','श','ष','स','ह','क्ष','त्र','ज्ञ','ग','घ','ङ','च','छ','ज','झ',
                 '०','१','२','३','४','५','६','७','८','९']
newmodel = load_model('Handwritten_OCR.h5')


app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))



def ROI(img):
    row, col = img.shape

    np_gray = np.array(img, np.uint8)
    one_row = np.zeros((1, col), np.uint8)

    images_location = []

    line_seg_img = np.array([])
    for r in range(row - 1):
        if np.equal(img[r:(r + 1)], one_row).all():
            if line_seg_img.size == 0:
                current_r = r
            else:
                images_location.append(line_seg_img[:-1])
                line_seg_img = np.array([])
                current_r = r
        else:
                #             print(r)
            if line_seg_img.size <= 1:
                line_seg_img = np.vstack((np_gray[r], np_gray[r + 1]))

            else:
                line_seg_img = np.vstack((line_seg_img, np_gray[r + 1]))

    return images_location


def preprocessing(img):
        # resizing the image
    img = cv.resize(img, (800, 600), interpolation=cv.INTER_AREA)
    image_area = img.shape[0] * img.shape[1]

        #     converting into grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gaussian = cv.GaussianBlur(img_gray, (3, 3), 0)
    _, thresh_img = cv.threshold(gaussian, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #     dilated = cv.dilate(thresh_img, None, iterations=1)

        # finding the boundary of the all threshold images
    contours, _ = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #     print(len(contours))
    for contour in contours:
            # boundary of each contour
        x, y, w, h = cv.boundingRect(contour)
            # to discard very small noises
        if cv.contourArea(contour) < image_area * 0.0001:
            thresh_img[y:(y + h), x:(x + w)] = 0
        #     cv.imshow('img1',thresh_img)
        #     cv.drawContours(thresh_img, contours, -1, (255,0,0), 1)

        # line segmentation
    line_segmentation = ROI(thresh_img)

        # word segmentation
    each_word_segmentation = []
    for line in np.asarray(line_segmentation):
        word_segementation = ROI(line.T)
        for words in np.asarray(word_segementation):
            each_word_segmentation.append(words.T)

        #     cv.imshow('img',line_segmentation[1])

        #     cv.waitKey(0)
        #     cv.destroyAllWindows()
    return each_word_segmentation


def dikka_remove(output):
    resultafterdikka = []
    each_character = []

    for i in range(0, len(output)):
        each = []
        main = output[i]
        r, inv3 = cv.threshold(main, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        ig = output[i]
        row, col = ig.shape

                # Detect horizontal lines and removing largest line which is DIKA in word.
        horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
        detect_horizontal = cv.morphologyEx(ig, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
        cnts, _ = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if len(cnts) > 0:
            c = max(cnts, key=cv.contourArea)
            X, Y, w, h = cv.boundingRect(c)
            ig[0:Y + h + 2, 0:X + w].fill(0)

            r, inv1 = cv.threshold(ig, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            cnts1, _ = cv.findContours(inv1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for co in reversed(cnts1):
                if cv.contourArea(co) > 300:
                    X, Y, w, h = cv.boundingRect(co)
                    cv.rectangle(inv3, (X, 0), (X + w, Y + h), 255, 1)
                    each.append((inv3[0:Y + h, X:X + w]))
            each_character.append(each)
            resultafterdikka.append(inv3)

    return resultafterdikka, each_character



@app.route('/')
def upload_file():
    path = "static/outimg"
    willremoveimage = os.listdir(path)
    if not willremoveimage:
        pass
    else:
        for i in willremoveimage:
            os.remove(path+"/"+ i)

    return render_template('index.html',title='Devnagarik - Home')



@app.route('/send_pic',methods=['POST'])
def button_pressed():
    # print("Image recieved")
    dimensions = (100, 100)
    data_url = request.values['imgBase64']
    encoded_data = data_url.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)

    name = list(string.ascii_letters)
    word = preprocessing(img)
    for i in range(len(word)):
        cv.imwrite("static/outimg/image-" + name[i] + ".jpg", word[i])

    for i in range(len(word)):
        resize = cv.resize(word[i], (32, 32)) / 255.0;
        reshaped = np.reshape(resize, (1, 32, 32, 1))

        prediction = newmodel.predict(reshaped)
        max = prediction.argmax()
        predict_character = devnagarik_word[max]
        char=""
        char += predict_character

    return char


@app.route('/uploader', methods=['GET','POST'])
def upload():
    path = "static/outimg"
    willremoveimage = os.listdir(path)
    print(willremoveimage)
    for i in willremoveimage:
        os.remove(path+"/"+ i)

        

    target = os.path.join(APP_ROOT, 'static/images/')
    # print(target)
    # l=[]
    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST':
        # for file in request.files.getlist("file"):// for multiple file uplaod
        file= request.files['file']
        filename = file.filename
        destination = "/".join([target, filename])
        # print(destination)
        file.save(destination)
        newDes = os.path.join('static/images/' + filename)
        readingimg = cv.imread(newDes)




        def prediction(each_character):
            final_all_word = ""
            for i in range(len(each_character)):
                each_word = ""
                for j in each_character[i]:
                    character_img = j
                    resize = cv.resize(j, (32, 32)) / 255.0;
                    reshaped = np.reshape(resize, (1, 32, 32, 1))

                    prediction = newmodel.predict(reshaped)
                    max = prediction.argmax()
                    predict_character = devnagarik_word[max]
                    each_word += predict_character
                final_all_word += each_word + ' '
            return final_all_word

        output = preprocessing(readingimg)
        resultafterdikka, each_character= dikka_remove(output)
        final_result = prediction(each_character)





        makingimagename = list(string.ascii_letters)

        for i in range(len(resultafterdikka)):
            cv.imwrite("static/outimg/image-"+makingimagename[i]+".jpg", resultafterdikka[i])

        images = os.listdir("static/outimg")


    return render_template('index.html',photos=newDes,images_name = images,result=final_result,title='Devnagrik - Predict')



if __name__ == '__main__':
    app.run(debug=True)