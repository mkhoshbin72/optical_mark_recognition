import cv2
import numpy as np
import imutils
from imutils import contours
import argparse



def preprocess(img_path, gray=False):

    img = cv2.imread(img_path)
    if gray == False:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif gray == True:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def write_to_txt(choices, output_name):
    f = open(output_name, 'w')
    for i,c in enumerate(choices):
        f.write(f"{c}\n")

    f.close()


def make_bbox(points):

    xs = points[:,0]
    ys = points[:,1]

    top_left_x = min(xs)
    top_left_y = min(ys)
    bot_right_x = max(xs)
    bot_right_y = max(ys)

    return top_left_x, top_left_y, bot_right_x, bot_right_y



def calculate_score(correct_answers, choices, number_of_questions=80):

    answers = open(correct_answers)
    answers = answers.readlines()

    t=0
    f=0
    e=0
    b=0

    for i,a in enumerate(answers):
        a = int(a.split('\n')[0])

        if choices[i] == a:
            t+=1
        elif choices[i] == 'Bad':
            b+=1
        elif choices[i] != a and choices[i] != 0:
            f+=1
        elif choices[i] != a and choices[i] == 0:
            e+=1

    return (t-(f/3))/number_of_questions



def main(img_path):

    gray_img = preprocess(img_path, gray=True)

    blurred = cv2.GaussianBlur(gray_img, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    cnts = imutils.grab_contours(cnts)
    docCnt = None

    if len(cnts) > 0:

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts:

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                docCnt = approx
                break

    points = docCnt.reshape(4,2)

    top_left_x, top_left_y, bot_right_x, bot_right_y = make_bbox(points)

    croped = gray_img[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1]

    croped2 = cv2.GaussianBlur(croped.copy(), (3,3), 2)
    thresh = cv2.threshold(croped2, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    questionCnts = []

    for c in cnts:

        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if w >= 20 and h >= 20 and ar >= 1. and ar <= 2.:
            questionCnts.append(c)

    croped2 = croped.copy()
    c = 0
    sheet = np.zeros((4,300))
    ch=0
    s=0

    for j in range(0,1200,50):
        
        qc = contours.sort_contours(questionCnts,method="lef-to-right")[0]
        qc = contours.sort_contours(qc[j:j+50],method="top-to-bottom")[0]

        for ind,i in enumerate(qc):

            c+=1
            cnt = np.squeeze(i, 1)

            top_left_x, top_left_y, bot_right_x, bot_right_y = make_bbox(cnt)

            ans = cv2.threshold(croped2[top_left_y:bot_right_y+1, top_left_x:bot_right_x+1], 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            total = cv2.countNonZero(ans)

            if total > 300:
                sheet[ch, ind+(50*s)] = 1

        if ch < 3:
            ch+=1
        else:
            ch = 0
            s+=1

    choices = []
    for i in range(300):

        opt = np.where(sheet[:,i] == 1.0)[0]
        if len(opt) == 1:
            choice = opt[0]+1
        elif len(opt) == 0:
            choice = 0
        else:
            choice = 'BAD'

        choices.append(choice)

    return choices


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True,
                        help='Path to image')
    parser.add_argument('-o', '--out', required=False, default='out.txt',
                        help='output file name')
    parser.add_argument('-s', '--score', required=False, default=False, action='store_true',
                        help='Calculate the score')               
    
    args = parser.parse_args()

    img_path = args.image
    out = args.out
    s = args.score

    choices = main(img_path)

    write_to_txt(choices, out)

    if s == True:
        score = calculate_score('answers.txt', choices)
        print('Score: ', score*100)

