from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage import feature
from fast_slic import Slic
from skimage.filters import threshold_local
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage import io
import numpy as np
from functools import reduce
import math
import mahotas
import imutils
import argparse
import cv2
import random as rng
import time
rng.seed(12345)
def bluranje(image):
    def callback(x):
        pass
    cv2.namedWindow('Blur')

    color_min = 0
    space_min = 0
    diameter_min =0



    # create trackbars for color change
    low_k = 0  # slider start position
    high_k = 21

    cv2.createTrackbar('Kernel', 'Blur', low_k, high_k, callback)
    cv2.createTrackbar('Color', 'Blur', color_min, 300, callback)
    cv2.createTrackbar('Diameter', 'Blur', diameter_min, 40, callback)
    cv2.createTrackbar('Space', 'Blur', space_min, 300, callback)
    cv2.createTrackbar('Tip bluranja', 'Blur', 0, 4, callback)
    print("Odredi parametra za bluranje")


    while (True):
        global ukupno_vrijeme
        blur_type = cv2.getTrackbarPos('Tip bluranja', 'Blur')
        ksize = cv2.getTrackbarPos('Kernel', 'Blur')  # returns trackbar position
        if blur_type == 0:
            blurred= image
            cv2.imshow('Blur', blurred)

        if blur_type == 1:
            color_min = cv2.getTrackbarPos('Color', 'Blur')
            space_min = cv2.getTrackbarPos('Space', 'Blur')
            diameter_min = cv2.getTrackbarPos('Diameter', 'Blur')
            start1 = time.time()
            blurred = cv2.bilateralFilter(image, diameter_min, color_min, space_min)
            elapsed1 = time.time() - start1
            ukupno_vrijeme += elapsed1
            print('Izabrani tip bluranja: Bilateralni filter')
            cv2.imshow('Blur', blurred)
        if blur_type == 2:
            ksize = 2 * ksize + 1
            start2 = time.time()
            blurred = cv2.blur(image, (ksize,ksize))
            elapsed2 = time.time() - start2
            ukupno_vrijeme += elapsed2
            print('Izabrani tip bluranja: Average blur')
            cv2.imshow('Blur', blurred)
        if blur_type == 3:
            ksize = 2 * ksize + 1
            start3 = time.time()
            blurred = cv2.GaussianBlur(image, (ksize,ksize), 0)
            elapsed3 = time.time() - start3
            ukupno_vrijeme += elapsed3
            print('Izabrani tip bluranja: Gausian blur')
            cv2.imshow('Blur', blurred)
        if blur_type == 4:
            ksize = 2 * ksize + 1
            start4 = time.time()
            blurred = cv2.medianBlur(image, ksize)
            elapsed4 = time.time() - start4
            ukupno_vrijeme += elapsed4
            print('Izabrani tip bluranja: Median blur')
            cv2.imshow('Blur', blurred)





        k = cv2.waitKey(1000) & 0xFF  # large wait time to remove freezing
        if k == 32 :
            print("Bluranje je završeno")
            return blurred,color_min,space_min,diameter_min,ksize,blur_type
def morfanje(image):
    global ukupno_vrijeme


    def callback(x):
        pass

    cv2.namedWindow('Morpho')
    low_k = 0  # slider start position
    high_k = 21

    cv2.createTrackbar('Kernel', 'Morpho', low_k, high_k, callback)

    cv2.createTrackbar('Tip morphologije', 'Morpho', 0, 7, callback)


    while (True):
        # grab the frame
        ksize = cv2.getTrackbarPos('Kernel', 'Morpho')  # returns trackbar position



        # get trackbar positions

        morpho_type = cv2.getTrackbarPos('Tip morphologije', 'Morpho')
        if morpho_type == 0:
            print('Izabrani tip morphologije: None')
            morpho = image
            cv2.imshow('Morpho', image)

        if morpho_type == 1 :
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.erode(image, kernel, iterations=1)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Erodion')
            cv2.imshow('Morpho',morpho)



        if morpho_type == 2:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.dilate(image, kernel, iterations=1)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Dilation')
            cv2.imshow('Morpho',morpho)


        if morpho_type == 3:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Opening')
            cv2.imshow('Morpho',morpho)


        if morpho_type == 4:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Closing')
            cv2.imshow('Morpho',morpho)


        if morpho_type == 5:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Gradient')
            cv2.imshow('Morpho',morpho)


        if morpho_type == 6:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije: Blackhat')
            cv2.imshow('Morpho',morpho)
        if morpho_type == 7:
            ksize = 2 * ksize + 1
            kernel = np.ones((ksize, ksize), np.uint8)
            start = time.time()
            morpho = cv2.morphologyEx(image, cv2.MORPH_CROSS, kernel)
            elapsed = time.time() - start
            ukupno_vrijeme += elapsed
            print('Izabrani tip morphologije:Cross')
            cv2.imshow('Morpho',morpho)
        k = cv2.waitKey(1000) & 0xFF


        if k == 32 :
            print('Morfoloska obrada završena')


            return morpho, ksize,morpho_type
def biraj_verziju_slike(image):
    def callback(x):
        pass

    def daj_sliku(tip_slike):
        if tip_slike == 0:
            return image
        if tip_slike == 1:
            return rgb_image
        if tip_slike == 2:
            return gray_image
        if tip_slike == 3:
            return hsv_image
        if tip_slike == 4:
            return Lab_image


    cv2.namedWindow('Format')
    tip_slike_min = 0
    tip_slike_max = 4

    cv2.createTrackbar('Tip slike', 'Format', tip_slike_min, tip_slike_max, callback)


    while (True):
         # large wait time to remove freezing
        tip_slike = cv2.getTrackbarPos('Tip slike', 'Format')


        print("Izaberi tip sliku..")
        if tip_slike == 0:
            print("Izabran format slike: BGR")

            cv2.imshow('Format',image)
        if tip_slike == 1:
            print("Izabran format slike: RGB")
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Format',rgb_image)


        if tip_slike == 2:
            print("Izabran format slike: Grayscale")
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow('Format', gray_image)


        if tip_slike == 3:
            print("Izabran format slike: HSV")
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            cv2.imshow('Format', hsv_image)

        if tip_slike == 4:
            print("Izabran format slike: Lab")
            Lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
            cv2.imshow('Format', Lab_image)
        k = cv2.waitKey(1000) & 0xFF

        if k == 32:
            print("Izabran format slike ")
            return daj_sliku(tip_slike),tip_slike
def segment_colorfulness(image, mask):
    # split the image into its respective RGB components, then mask
    # each of the individual RGB channels so we can compute
    # statistics only for the masked region
    (B, G, R) = cv2.split(image.astype("float"))
    R = np.ma.masked_array(R, mask=mask)
    G = np.ma.masked_array(B, mask=mask)
    B = np.ma.masked_array(B, mask=mask)

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`,
    # then combine them
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)
ukupno_vrijeme = 0
def racunaj_segmente(image):
    global ukupno_vrijeme
    def callback(x):
        pass
    cv2.namedWindow('Postavi segmente')



    cv2.createTrackbar('Broj segmenata', 'Postavi segmente', 1, 1000, callback)
    cv2.createTrackbar('Compactness', 'Postavi segmente', 1, 10000, callback)
    cv2.createTrackbar('Min size', 'Postavi segmente', 1, 100, callback)

    print("Odredi parametra za bluranje")


    while (True):

        cv2.imshow('Postavi segmente', image)
        cv2.waitKey(10000000)
        broj_segmenata=cv2.getTrackbarPos('Broj segmenata', 'Postavi segmente')
        compac = cv2.getTrackbarPos('Compactness', 'Postavi segmente')
        compac = compac/100
        min_size = cv2.getTrackbarPos('Min size', 'Postavi segmente')
        start1 = time.time()
        segments = Slic(image, n_segments=broj_segmenata, compactness=compac,min_size_factor=min_size)
        elapsed1 = time.time() - start1
        ukupno_vrijeme += elapsed1
        fig = plt.figure("Superpixels")
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments))
        plt.axis("off")
        plt.show()
        k = cv2.waitKey(1) & 0xFF  # large wait time to remove freezing
        if k == 32:
            print("Superpixeli podeseni ")
            return segments
def racunaj_karakteristike2(lista,assignment):
    brojac3 = 0
    def racunaj_texturu(lista):
        mask = np.zeros(image.shape[:2],dtype='uint8')
        redni = int(lista['number'])
        mask[assignment == redni] = 255
        gray = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        Mi = cv2.moments(mask)
        cX = int(Mi["m10"] / Mi["m00"])
        cY = int(Mi["m01"] / Mi["m00"])
        C = segment_colorfulness(image,mask=mask)
        c_unit8 = cv2.bitwise_and(image, image, mask=mask)
        hist_text = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
        lista.update({'textura': hist_text})
        lista.update({'kontura': mask})
        lista.update({'color': C})
        lista.update({'yx': (cY,cX)})
        return lista
    lista_klastera= list(map(racunaj_texturu, lista))
def racunaj_karakteristike(segments,image):
    suma_kontura = np.zeros(image.shape[:2], dtype='uint8')
    brojac3 = 0
    for v in np.unique(segments):

        mask = np.zeros(image.shape[:2])
        mask[segments == v] = 255
        mask = mask.astype('uint8')
        gray = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        maska_za_bijele = np.zeros(gray.shape[:2])
        maska_za_ostatke = np.zeros(gray.shape[:2])
        maska_za_bijele = np.where(gray == [255],255,0)
        maska_za_bijele = maska_za_bijele.astype('uint8')
        maska_za_ostatke = np.where( ([255] > gray) & (gray > [0]),255,0)
        maska_za_ostatke = maska_za_ostatke.astype('uint8')
        brojac1 = len((np.where(maska_za_bijele == [255]))[0])
        brojac2 = len((np.where(maska_za_ostatke == [255]))[0])
        if brojac1!=0:
            print("Kontura {} u sebi ima bijele boje, treba provjeriti jel potpuno bijela ili djelimicno".format(v))
            if brojac2 / brojac1 > 0.2:
                print("Konturu {} treba razdvojiti na dvije : bijelu i ostatak".format(v))
                suma_kontura = cv2.bitwise_or(suma_kontura, maska_za_bijele)
                cnt_ostatak1, h = cv2.findContours(maska_za_ostatke, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cnt_ostatak1 = sorted(cnt_ostatak1, key=cv2.contourArea, reverse=True)
                if len(cnt_ostatak1)>1:

                    for i,c1 in enumerate(cnt_ostatak1):
                        maska_za_konturu_ostataka = np.zeros(gray.shape[:2],dtype='uint8')
                        cv2.drawContours(maska_za_konturu_ostataka, [c1], -1, (255,255,255), -1)
                        brojac4 = len((np.where(maska_za_konturu_ostataka == [255]))[0])
                        if brojac4 < 10 or cv2.contourArea(c1)< 0.005*cv2.contourArea(cnt_ostatak1[0]):
                            suma_kontura = cv2.bitwise_or(suma_kontura, maska_za_konturu_ostataka)
                            image[maska_za_konturu_ostataka == 255] = (255, 255, 255)

                        else:
                            Mi = cv2.moments(maska_za_konturu_ostataka)
                            cXi = int(Mi["m10"] / Mi["m00"])
                            cYi = int(Mi["m01"] / Mi["m00"])
                            C_c1 = segment_colorfulness(image, maska_za_konturu_ostataka)
                            c_unit8 = cv2.bitwise_and(image, image, mask=maska_za_konturu_ostataka)
                            #hist2_text_c1=mahotas.features.haralick(c_unit8).mean(axis=0)
                            hist_text_c1 = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
                            karakteristika = {1: brojac3 + 1, 2: C_c1, 3: hist_text_c1, 4: (cXi, cYi), 5: c1,
                                              6: maska_za_konturu_ostataka}
                            lista_ostataka.append(karakteristika)
                            brojac3 = brojac3 + 1


                elif len(cnt_ostatak1)==1:
                    brojac4 = len((np.where(maska_za_ostatke == [255]))[0])
                    if brojac4 < 15:
                        suma_kontura = cv2.bitwise_or(suma_kontura, maska_za_ostatke)
                        image[maska_za_ostatke == 255] = (255, 255, 255)
                    else :
                        Mi = cv2.moments(maska_za_ostatke)
                        cX = int(Mi["m10"] / Mi["m00"])
                        cY = int(Mi["m01"] / Mi["m00"])
                        C = segment_colorfulness(image, maska_za_ostatke)
                        c_unit8 = cv2.bitwise_and(image, image, mask=maska_za_ostatke)
                        #hist2_text_c1 = mahotas.features.haralick(c_unit8).mean(axis=0)
                        hist_text = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
                        karakteristika = {1: brojac3 + 1, 2: C, 3: hist_text, 4: (cX, cY), 5: cnt_ostatak1,
                                          6: maska_za_ostatke}
                        lista_ostataka.append(karakteristika)
                        brojac3 = brojac3 + 1

                print("Razdvajanje je završeno")

            else:
                print("Napravi cistu bijelu konturu {}".format(v))
                suma_kontura =  cv2.bitwise_or(suma_kontura,maska_za_bijele)



        else:
            print("Kontura {} nema u sebi bijele boje".format(v))
            cnt_bez_bijele, h = cv2.findContours(maska_za_ostatke,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cnt_bez_bijele = sorted(cnt_bez_bijele, key=cv2.contourArea, reverse=True)
            if len(cnt_bez_bijele) > 1:

                for i, c2 in enumerate(cnt_bez_bijele):
                    maska_za_konturu_ostataka3 = np.zeros(gray.shape[:2], dtype='uint8')
                    cv2.drawContours(maska_za_konturu_ostataka3, [c2], -1, (255, 255, 255), -1)
                    brojac4 = len((np.where(maska_za_konturu_ostataka3 == [255]))[0])
                    if brojac4 < 15 or  cv2.contourArea(c2)< 0.005*cv2.contourArea(cnt_bez_bijele[0]):
                        suma_kontura = cv2.bitwise_or(suma_kontura, maska_za_konturu_ostataka3)
                        image[maska_za_konturu_ostataka3 == 255] = (255, 255, 255)
                    else:
                        Mi = cv2.moments(maska_za_konturu_ostataka3)
                        cXi = int(Mi["m10"] / Mi["m00"])
                        cYi = int(Mi["m01"] / Mi["m00"])
                        C = segment_colorfulness(image, maska_za_konturu_ostataka3)
                        c_unit8 = cv2.bitwise_and(image, image, mask=maska_za_konturu_ostataka3)
                        #hist2_text_c1 = mahotas.features.haralick(c_unit8).mean(axis=0)
                        hist_text = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
                        karakteristika = {1: brojac3 + 1, 2: C, 3: hist_text, 4: (cXi, cYi), 5: c2,
                                          6: maska_za_konturu_ostataka3}
                        lista_ostataka.append(karakteristika)
                        brojac3 = brojac3 + 1


            elif len(cnt_bez_bijele) == 1:
                maska_za_konturu_ostataka4 = np.zeros(gray.shape[:2], dtype='uint8')
                cv2.drawContours(maska_za_konturu_ostataka4, cnt_bez_bijele, -1, (255, 255, 255), -1)
                brojac4 = len((np.where(maska_za_konturu_ostataka4 == [255]))[0])
                if brojac4 < 15 :
                    suma_kontura = cv2.bitwise_or(suma_kontura, maska_za_konturu_ostataka4)
                    image[maska_za_konturu_ostataka4 == 255] = (255, 255, 255)
                else:
                    Mi = cv2.moments(maska_za_konturu_ostataka4)
                    cX = int(Mi["m10"] / Mi["m00"])
                    cY = int(Mi["m01"] / Mi["m00"])
                    C = segment_colorfulness(image, maska_za_konturu_ostataka4)
                    c_unit8 = cv2.bitwise_and(image, image, mask=maska_za_konturu_ostataka4)
                    #hist2_text_c1 = mahotas.features.haralick(c_unit8).mean(axis=0)
                    hist_text = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
                    karakteristika = {1: brojac3 + 1, 2: C, 3: hist_text, 4: (cX, cY), 5: cnt_bez_bijele,
                                      6: maska_za_konturu_ostataka4}
                    lista_ostataka.append(karakteristika)
                    brojac3 = brojac3 + 1


            print("Ostavi konturu")


    print("broj  kontura  je :{}".format(len(lista_ostataka)))
    cv2.imshow('bijela',suma_kontura)
    cv2.waitKey(0)
    if brojac1 !=0:
        cv2.imshow('bijela', suma_kontura)
        cv2.waitKey(0)
        Mi = cv2.moments(suma_kontura)
        cX = int(Mi["m10"] / Mi["m00"])
        cY = int(Mi["m01"] / Mi["m00"])
        C = 255

        testna_nula = cv2.bitwise_and(image, image, mask=suma_kontura)

        hist_text_svih_nula = desc.describe(cv2.cvtColor(testna_nula, cv2.COLOR_BGR2GRAY))
        karakteristika_nula = {1: 0, 2: C, 3: hist_text_svih_nula, 4: (cX, cY), 5: 0, 6: suma_kontura}
        lista_karakteristika.append(karakteristika_nula)

def detekcija_ivica(image):
    def callback(x):
        pass


    cv2.namedWindow('Detekcija ivica')
    low_k = 0  # slider start position
    high_k = 255


    cv2.createTrackbar('Min. treshold', 'Detekcija ivica', low_k, high_k, callback)
    cv2.createTrackbar('Max. treshold', 'Detekcija ivica', low_k, high_k, callback)
    cv2.createTrackbar('L2gradient','Detekcija ivica',0,1,callback)
    cv2.createTrackbar('aperture', 'Detekcija ivica', 0, 2, callback)
    cv2.createTrackbar('edge_type','Detekcija ivica',0,6,callback)
    cv2.createTrackbar('a', 'Detekcija ivica', 0, 40, callback)
    cv2.createTrackbar('Blocksize', 'Detekcija ivica', 1, 21, callback)
    cv2.createTrackbar('Sigma','Detekcija ivica',0,50,callback)

    while (True):
        global ukupno_vrijeme
        # grab the frame
        edge_type = cv2.getTrackbarPos('edge_type','Detekcija ivica')
        min = cv2.getTrackbarPos('Min. treshold', 'Detekcija ivica')
        max = cv2.getTrackbarPos('Max. treshold', 'Detekcija ivica')
        gradient = cv2.getTrackbarPos('L2gradient', 'Detekcija ivica')
        aperture = cv2.getTrackbarPos('aperture', 'Detekcija ivica')
        blocksize = cv2.getTrackbarPos('Blocksize', 'Detekcija ivica')
        sigma = cv2.getTrackbarPos('Sigma', 'Detekcija ivica')
        a = cv2.getTrackbarPos('a', 'Detekcija ivica')
        if edge_type == 0:
            print('Izabrani tip edge detekcije: None')
            cv2.imshow('Detekcija ivica', image)
            ivica = image

        if edge_type == 1:
            aperture = 2*aperture+3
            print('Izabrani tip edge detekcije: Canny')
            if gradient == 0:
                start = time.time()
                ivica = cv2.Canny(image, min, max, apertureSize=aperture, L2gradient=False)
                elapsed = time.time() - start
                ukupno_vrijeme += elapsed
                cv2.imshow('Detekcija ivica', ivica)
            if gradient == 1 :
                start1 = time.time()
                ivica = cv2.Canny(image, min, max, apertureSize=aperture, L2gradient=True)
                elapsed1 = time.time() - start1
                ukupno_vrijeme += elapsed1
                cv2.imshow('Detekcija ivica', ivica)
        if edge_type == 2:
            print('Izabrani tip edge detekcije: Laplacian')

            start2 = time.time()
            ivica = cv2.Laplacian(image, cv2.CV_64F)
            ivica = cv2.convertScaleAbs(ivica)
            elapsed2 = time.time() - start2
            ukupno_vrijeme += elapsed2
            cv2.imshow('Detekcija ivica', ivica)
        if edge_type == 3:
            print('Izabrani tip edge detekcije: Sobel')
            start3 = time.time()
            gX = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=1, dy=0)
            gY = cv2.Sobel(image, ddepth=cv2.CV_64F, dx=0, dy=1)

            # the `gX` and `gY` images are now of the floating point data type,
            # so we need to take care to convert them back a to unsigned 8-bit
            # integer representation so other OpenCV functions can utilize them
            gX = cv2.convertScaleAbs(gX)
            gY = cv2.convertScaleAbs(gY)

            # combine the sobel X and Y representations into a single image
            ivica = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
            elapsed3 = time.time() - start3
            ukupno_vrijeme += elapsed3

            # show our output images

            cv2.imshow("Detekcija ivica", ivica)
        if edge_type == 4:
            print('Izabrani tip edge detekcije: Filter2D')
            a = a-20
            #kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
            kernel = np.array([[1, 1, 1], [1, a, 1], [1, 1, 1]], dtype=np.float32)
            start4 = time.time()
            Laplacian = cv2.filter2D(image, cv2.CV_32F, kernel)
            elapsed4 = time.time() - start4
            ukupno_vrijeme += elapsed4
            sharp = np.float32(image)
            imgResult = sharp - Laplacian
            # convert back to 8bits gray scale
            imgResult = np.clip(imgResult, 0, 255)
            ivica = imgResult.astype('uint8')
            cv2.imshow('Detekcija ivica', ivica)
        if edge_type == 5:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            print('Izabrani tip edge detekcije: adaptive threshold')
            blocksize = 2*blocksize+1
            start5 = time.time()
            thresh = cv2.adaptiveThreshold(gray, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blocksize, 15)
            elapsed5 = time.time() - start5
            ukupno_vrijeme += elapsed5
            T = threshold_local(gray, blocksize, offset=5, method="gaussian")
            ivica = (gray < T).astype("uint8") * 255
            cv2.imshow("Detekcija ivica", ivica)
        if edge_type == 6:
            sigma = 0.1*float(sigma)
            print('Izabrani tip edge detekcije: Auto canny')
            start6 = time.time()
            ivica = imutils.auto_canny(image,sigma=sigma)
            elapsed6 = time.time() - start6
            ukupno_vrijeme += elapsed6
            cv2.imshow("Detekcija ivica", ivica)
        if edge_type == 7:
            print('Izabrani tip edge detekcije: obicni threshold')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            start7 = time.time()
            (T, ivica) = cv2.threshold(gray,min,max, cv2.THRESH_BINARY_INV)
            elapsed7 = time.time() - start7
            ukupno_vrijeme += elapsed7
            cv2.imshow("Detekcija ivica", ivica)
        k = cv2.waitKey(1000) & 0xFF  # large wait time to remove freezing
        if k == 32:
            print("Izabran edge metoda ")
            return ivica,min,max,aperture,gradient,blocksize,sigma,a,edge_type
def distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# za traženje razlike u bojama
def razlika_boja(p1,p2):
	return 10*abs(p1-p2)

# za trazenje razlike u texturi :
def razlika_textura(p1,p2):
	return 100 * np.sum(((p1 - p2) ** 2) / (p2 + p1 + 1e-10))

# za traženje texturnih karakteristika segmenta:
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# compute the Local Binary Pattern representation of the image, and then
		# use the LBP representation to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(), bins=range(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# return the histogram of Local Binary Patterns
		return hist

def funkcija2(lista):
    print("Pristupa se provjeri segmenata za spajanje")
    nova_lista = []
    brojac = 0

    while(brojac<len(lista)):

        for i in enumerate(lista):

            print("{} segment se ispituje".format(i[1][1]))
            if ima_slicnu2(i[1],lista) !=0: #ako ima slicnu
                sazimaj2(i[1],ima_slicnu2(i[1],lista), lista,nova_lista)
            else:
                print("{} segment nije pronasao sebi za spajanje".format(i[1][1]))
                brojac += 1

                nova_lista.append(i[1])

            if brojac == len(lista):
                break
            else:
                if i == len(lista)-1:
                    i = 0
                    brojac = 0
    print("Nova lista je spremna")
    return nova_lista
def funkcija(lista):
    print("Pristupa se provjeri segmenata za spajanje")
    nova_lista = []
    brojac = 0

    while(brojac<len(lista)):

        for i in enumerate(lista):
            redni = i[1]['number']

            print("{} segment se ispituje".format(redni))
            if ima_slicnu(i[1],lista) !=0: #ako ima slicnu
                sazimaj(i[1],ima_slicnu(i[1],lista), lista)
            else:
                print("{} segment nije pronasao sebi za spajanje".format(redni))
                brojac += 1



            if brojac == len(lista):
                break
            else:
                if i == len(lista)-1:
                    i = 0
                    brojac = 0
    print("Nova lista je spremna")
    return lista
def sazimaj(a,b, lista):
    redni1=a['number']
    redni2 = b['number']
    lista_komponenti = []
    lista_komponenti.append(redni1)
    lista_komponenti.append(redni2)
    mask1 = np.zeros(image.shape[:2], dtype='uint8')
    mask2 = np.zeros(image.shape[:2], dtype='uint8')
    mask1[assignment == redni1] = 255
    mask2[assignment == redni2] = 255
    maska_nova=cv2.bitwise_or(mask1, mask2)
    cv2.imshow('maska',maska_nova)
    cv2.waitKey(0)
    cnt, h = cv2.findContours(maska_nova, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Mi = cv2.moments(maska_nova)
    cX = int(Mi["m10"] / Mi["m00"])
    cY = int(Mi["m01"] / Mi["m00"])
    a['number'] = lista_komponenti
    c_unit8 = cv2.bitwise_and(image, image, mask=maska_nova)
    C_novi = segment_colorfulness(image, mask=maska_nova)
    hist_text_novi = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
    a['textura'] = hist_text_novi
    a['yx'] = (cY, cX)
    a['kontura'] = maska_nova
    a['color'] = C_novi



    lista.remove(b)
def sazimaj2(a,b, lista, nova_lista):
    lista_komponenti = []
    lista_komponenti.append(a[1])
    lista_komponenti.append(b[1])
    maska_nova=cv2.bitwise_or(a[6], b[6])
    cnt, h = cv2.findContours(maska_nova, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    Mi = cv2.moments(maska_nova)
    cX = int(Mi["m10"] / Mi["m00"])
    cY = int(Mi["m01"] / Mi["m00"])
    a[1] = lista_komponenti
    c_unit8 = cv2.bitwise_and(image, image, mask=maska_nova)
    hist_text_novi = desc.describe(cv2.cvtColor(c_unit8, cv2.COLOR_BGR2GRAY))
    a[3] = hist_text_novi
    a[2] = segment_colorfulness(image, mask=maska_nova)
    a[4] = (cX, cY)
    a[5] = cnt
    a[6] = maska_nova

    lista.remove(b)
    nova_lista.append(a)
def ima_slicnu(i,lista):
    redni1 = i['number']
    centroid1=i['yx']
    textura1 =i['textura']
    boja1=i['color']
    slicni=0

    def lista_boja(lista):
        return 10 * abs(boja1 - lista['color']), lista['number']

    def lista_textura(lista):
        return 100 * np.sum(((textura1 - lista['textura']) ** 2) / (lista['textura'] + textura1 + 1e-10)), lista[
            'number']

    def lista_distanci(lista):
        return math.sqrt((centroid1[1] - lista['yx'][1]) ** 2 + (centroid1[0] - lista['yx'][0]) ** 2), lista['number']

    lista_distanca = sorted(list(map(lista_distanci, lista)))
    lista_textura = sorted(list(map(lista_textura, lista)))
    lista_boja = sorted(list(map(lista_boja, lista)))
    prag_udaljenosti = lista_distanca[1][0]
    prag_boja = lista_boja[1][0]
    prag_texture = lista_textura[1][0]
    for klaster in lista:
        redni2 = klaster['number']
        if redni1 != redni2:
            # cv2.imshow("segment {}".format(j),cv2.bitwise_and(image,image,mask= k[6]))
            # cv2.waitKey(0)

            distanca = distance(centroid1, klaster['yx'])
            print("Distanca izmedju {} segmenta i {} segmenta iznosi : {}".format(redni1, redni2, distanca))
            razlikab = razlika_boja(boja1, klaster['color'])
            print("Razlika u boji izmedju {} segmenta i {} segmenta iznosi : {}".format(redni1, redni2, razlikab))
            razlikat = razlika_textura(textura1,klaster['textura'])
            print("Razlika u texturi {} segmenta i {} segmenta iznosi : {}".format(redni1, redni2, razlikat))
            if distanca == lista_distanca[1][0] :
                if razlikat <= lista_textura[int(len(lista_textura)/3)][0] and razlikab <= lista_boja[int(len(lista_boja)/2)][0]:
                    print("{} segment je nasao slicnan segment, a to je segment {}".format(redni1, redni2))
                    prag_udaljenosti = distanca
                    prag_texture = razlikat
                    prag_boja = razlikab
                    slicni = klaster



            if razlikat == lista_textura[1][0]:
                if  distanca   <= lista_distanca[3][0] and  razlikab <= lista_boja[int(len(lista_boja)/2)][0]:
                    print("{} segment je nasao slicnan segment, a to je segment {}".format(redni1, redni2))
                    prag_udaljenosti = distanca
                    prag_texture = razlikat
                    prag_boja = razlikab
                    slicni = klaster

            if razlikat <= lista_textura[5][0] and  distanca <= lista_distanca[3][0] and  razlikab <=lista_boja[6][0]:
                    print("{} segment je nasao slicnan segment, a to je segment {}".format(redni1, redni2))
                    prag_udaljenosti = distanca
                    prag_texture = razlikat
                    prag_boja = razlikab
                    slicni = klaster

            else:
                continue

    return slicni
def ima_slicnu2(i,lista):
    prag_udaljenosti = 70
    koeficijent_udaljenost = 0.9
    koeficijent_boja = 0.9
    koeficijent_texura = 0.9
    prag_boja = 1
    slicni = 0
    prag_texture = 1
    for j,k  in enumerate(lista):


        if i[1] ==k[1]:
            print("Isti segment preskoci")
            continue
        if i[1]!=k[1]:
            #cv2.imshow("segment {}".format(j),cv2.bitwise_and(image,image,mask= k[6]))
            #cv2.waitKey(0)

            distanca = distance(i[4],k[4])
            print("Distanca izmedju {} segmenta i {} segmenta iznosi : {}".format(i[1],k[1], distanca))
            razlikab= razlika_boja(i[2],k[2])
            print("Razlika u boji izmedju {} segmenta i {} segmenta iznosi : {}".format(i[1], k[1], razlikab))
            razlikat = razlika_textura(i[3],k[3])
            print("Razlika u texturi {} segmenta i {} segmenta iznosi : {}".format(i[1], k[1], razlikat))
            if   distanca <=koeficijent_udaljenost*prag_udaljenosti and razlikat <=koeficijent_texura*prag_texture and razlikab<=koeficijent_boja*prag_boja:
                print("{} segment je nasao slicnan segment, a to je segment {}".format(i[1],k[1]))
                prag_udaljenosti = distanca
                prag_texture = razlikat
                prag_boja = razlikab
                slicni = k

            else:
                continue



    return slicni
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-s", "--segments", type=int, default=100,
	help="# of superpixels")
args = vars(ap.parse_args())
desc = LocalBinaryPatterns(24, 3)
nova_lista = []
color_min = 0
space_min = 0
tipmorfo = 0
tipmorfo2 = 0
diameter_min =0
mintres= 0
maxtres = 0
ksizeb = 0
ksizeb1= 0
ksizem= 0
ksizem1 = 0
ksizem2 = 0
broj_slike = args["image"]
broj_slike=int(''.join([n for n in broj_slike if n.isdigit()]))
image = io.imread(args["image"])
image = imutils.resize(image, width= 600)
verzija,tip_slike = biraj_verziju_slike(image)
blur, color_min, space_min, diameter_min,ksizeb,blur_type = bluranje(verzija)
cv2.waitKey(0)
canny, mintres, maxtres, aperture, gradient,blocksize,sigma,a,edge_type = detekcija_ivica(blur)
cv2.waitKey(0)
inputz = input("Da li ima zbunjastih poljana: ")
valz = int(inputz)
if valz == 0:
    dilate, ksizem, tipmorfo = morfanje(canny)
    cv2.waitKey(0)
    dilate2, ksizem1, tipmorfo2 = morfanje(dilate)
    cv2.waitKey(0)
if valz == 1:
    dilate, ksizem, tipmorfo = morfanje(canny)
    cv2.waitKey(0)
    blur1, color_min1, space_min1, diameter_min1,ksizeb1,blur_type1 = bluranje(dilate)
    cv2.waitKey(0)
    canny1, mintres1, maxtres1, aperture1, gradient1,blocksize1,sigma1,a1,edge_type1 = detekcija_ivica(blur1)
    cv2.waitKey(0)
    dilate1, ksizem1, tipmorfo2 = morfanje(canny1)
    cv2.waitKey(0)
    dilate2, ksizem2, tipmorfo3 = morfanje(dilate1)
    cv2.waitKey(0)


input1 = input("Da li ce ima bijelih granica: ")
val = int(input1)
if val == 0:

    testna,_ = biraj_verziju_slike(image)
    testna[np.where(dilate2 > 245)] = (255, 255, 255)
    cv2.imshow('image', testna)
    cv2.waitKey(0)
    input2 = input("Unesi stvaran broj kontura: ")
    val1 = int(input2)
    # default vrijednosti udaljenosti 0.80, boje 0.9, texture 0.8
    print("Racunaju se segmenti..")
    start1 = time.time()
    # segments, min_size, max_size, connectivity, compac, slic_zero = racunaj_segmente(testna)
    segments = slic(img_as_float(testna), n_segments=50, compactness=0.1, max_iter=10, sigma=0,
                    spacing=None, multichannel=True, convert2lab=None, enforce_connectivity=True, min_size_factor=3,
                    max_size_factor=9, slic_zero=False)
    elapsed1 = time.time() - start1
    ukupno_vrijeme += elapsed1
    print("Broj segmenata : {}".format(len(np.unique(segments))))
    desc = LocalBinaryPatterns(8, 1)
    lista_ostataka = []
    lista_karakteristika = []
    print("Racunaju se karakteristike segmenata..")
    start2 = time.time()
    racunaj_karakteristike(segments, testna)
    elapsed2 = time.time() - start2
    ukupno_vrijeme += elapsed2
    print("Karakteristike su izračunate")

    colors = []
    for i in enumerate(lista_ostataka):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    clone1 = image.copy()
    for i, kontura in enumerate(lista_ostataka):
        contours1, hierarchy1 = cv2.findContours(kontura[6],
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(clone1, contours1, -1, colors[i], -1)
        cv2.putText(clone1, "{}".format(kontura[1]), (kontura[4][0], kontura[4][1]), cv2.FONT_HERSHEY_SIMPLEX, .3,
                    (255, 255, 255))

    clone1[np.where(lista_karakteristika[0][6] == 255)] = (255, 255, 255)
    cv2.imshow('Trenutni segmenti', clone1)
    cv2.imwrite('Finalne_slike/trenutna{}.jpg'.format(broj_slike), clone1)
    cv2.waitKey(0)
    start3 = time.time()
    nova_lista = funkcija2(lista_ostataka)
    elapsed3 = time.time() - start3
    ukupno_vrijeme += elapsed3
    print("Izracunati broj spojenih segmenata : {}".format(len(nova_lista)))
    colors = []
    for i in enumerate(nova_lista):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    clone = image.copy()
    for i, kontura2 in enumerate(nova_lista):
        contours2, hierarchy1 = cv2.findContours(kontura2[6],
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(clone, contours2, -1, colors[i], -1)
        cv2.putText(clone, "{}".format(kontura2[1]), (kontura2[4][0], kontura2[4][1]), cv2.FONT_HERSHEY_SIMPLEX, .3,
                    (255, 255, 255))

        # --- Copy pixel values of logo image to room image wherever the mask is white ---

    clone[np.where(lista_karakteristika[0][6] == 255)] = (255, 255, 255)
    print("Procenat pogodjenih kontura: {} % ".format((len(nova_lista) / val1) * 100))

    cv2.imshow('Zavrsna slika nakon provjere segmenata', clone)

    cv2.imwrite('Finalne_slike/finalna{}.jpg'.format(broj_slike), clone)
    cv2.waitKey(0)

if val == 1 :
    input2 = input("Unesi stvaran broj kontura: ")
    val1 = int(input2)
    print("Racunaju se segmenti..")
    start1 = time.time()

    slic = Slic(num_components=100, min_size_factor=0, compactness=10)
    clone = image.copy()
    assignment = slic.iterate(image)
    lista_klastera = slic.slic_model.clusters
    print("Racunaju se karakteristike segmenata..")

    racunaj_karakteristike2(lista_klastera, assignment)
    print("Karakteristike su izračunate")
    fig = plt.figure("Superpixels")
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), assignment))
    plt.axis("off")
    plt.show()
    print("Broj segmenata : {}".format(len(slic.slic_model.clusters)))

    colors = []
    for i in enumerate(lista_klastera):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    clone1 = image.copy()
    for i, klaster in enumerate(lista_klastera):
        contours1, hierarchy1 = cv2.findContours(klaster['kontura'],
                                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(clone1, contours1, -1, colors[i], -1)
        cv2.putText(clone1, "{}".format(klaster['number']), (int(klaster['yx'][1]), int(klaster['yx'][0])),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

    cv2.imshow('Trenutni segmenti', clone1)
    cv2.imwrite('Finalne_slike/trenutna{}.jpg'.format(broj_slike), clone1)
    cv2.waitKey(0)

    nova_lista = funkcija(lista_klastera)
    print("Izracunati broj spojenih segmenata : {}".format(len(nova_lista)))
    colors = []
    for i in enumerate(nova_lista):
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))
    clone = image.copy()
    for i, klaster in enumerate(nova_lista):
        contours1, hierarchy1 = cv2.findContours(klaster['kontura'], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(clone, contours1, -1, colors[i], -1)
        cv2.putText(clone, "{}".format(klaster['number']), (int(klaster['yx'][1]), int(klaster['yx'][0])),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255))
    cv2.imshow('Zavrsna slika nakon provjere segmenata', clone)
    cv2.imwrite('Finalne_slike/finalna{}.jpg'.format(broj_slike), clone)
    cv2.waitKey(0)

if valz == 0 and val == 0:
    file = open("Finalne_slike/podacislike{}.txt".format(broj_slike), "w")
    L1 = ["Parametri prvog bluranja : Diameter: {} , Color: {} , Space : {} , Kernel: {} , Tip bluranja: {} \n".format(diameter_min, color_min, space_min,ksizeb,blur_type)]
    L2 = ["Parametri prvog odredjivanja ivica : Min.treshold:{},Max.treshold:{},Aperture:{},Gradient:{},Blocksize:{},Sigma:{},a:{},Edge metoda: {} \n".format(mintres, maxtres, aperture, gradient,blocksize,sigma,a, edge_type )]
    L3 = ["Da li ima zbunjastih poljana : NE \n"]
    L4 = ["Parametri prve morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem, tipmorfo)]
    L5 = ["Parametri druge morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem1, tipmorfo2)]
    L6 = ["Broj superpixela: {} \n".format(len(np.unique(segments)))]
    L7 = ["Preciznost:{} \n".format((len(nova_lista) / val1) * 100)]
    #L8 = ["Koeficijenti provjere: Udaljenost: {}, Boja: {}, Textura: {} \n".format(koeficijent_udaljenosti,koeficijent_boje,koeficijent_texure)]
    L8 = ["Vrijeme izvršavanja koda: {} \n".format(ukupno_vrijeme)]
    L9 = ["Tip slike: {} \n".format(tip_slike)]
    # \n is placed to indicate EOL (End of Line)
    file.write("Podaci slike {}: \n".format(broj_slike))
    file.writelines(L1)
    file.writelines(L2)
    file.writelines(L3)
    file.writelines(L4)
    file.writelines(L5)
    file.writelines(L6)
    file.writelines(L7)
    file.writelines(L8)
    file.writelines(L9)
    file.close()
if valz==1 and val == 0:
    file = open("Finalne_slike/podacislike{}.txt".format(broj_slike), "w")
    L1 = ["Parametri prvog bluranja : Diameter: {} , Color: {} , Space : {} , Kernel: {} , Tip bluranja: {} \n".format(
        diameter_min, color_min, space_min, ksizeb, blur_type)]
    L2 = ["Parametri prvog odredjivanja ivica : Min.treshold:{},Max.treshold:{},Aperture:{},Gradient:{},Blocksize:{},Sigma:{},a1:{},Edge metoda: {} \n".format(mintres, maxtres, aperture, gradient, blocksize, sigma, a, edge_type)]
    L3 = ["Da li ima zbunjastih poljana : DA \n"]
    L4 = ["Parametri prve morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem, tipmorfo)]
    L5 = ["Parametri drugog bluranja : Diameter: {} , Color: {} , Space : {} , Kernel: {} , Tip bluranja: {} \n".format(diameter_min1, color_min1, space_min1, ksizeb1, blur_type1)]
    L6 = ["Parametri drugog odredjivanja ivica : Min.treshold:{},Max.treshold:{},Aperture:{},Gradient:{},Blocksize:{},Sigma:{},a1:{},Edge metoda: {} \n".format(mintres1, maxtres1, aperture1, gradient1, blocksize1, sigma1, a1, edge_type1)]
    L7 = ["Parametri druge morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem1, tipmorfo2)]
    L8 = ["Parametri treće morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem2, tipmorfo3)]
    L9 = ["Broj superpixela: {} \n".format(len(np.unique(segments)))]
    L10 = ["Preciznost:{} \n".format((len(nova_lista) / val1) * 100)]
    #L11 = ["Koeficijenti provjere: Udaljenost: {}, Boja: {}, Textura: {} \n".format(koeficijent_udaljenosti,koeficijent_boje,koeficijent_texure)]
    L11 = ["Vrijeme izvršavanja koda: {} \n".format(ukupno_vrijeme)]
    L12 = ["Tip slike: {} \n".format(tip_slike)]
    # \n is placed to indicate EOL (End of Line)
    file.write("Podaci slike {}: \n".format(broj_slike))
    file.writelines(L1)
    file.writelines(L2)
    file.writelines(L3)
    file.writelines(L4)
    file.writelines(L5)
    file.writelines(L6)
    file.writelines(L7)
    file.writelines(L8)
    file.writelines(L9)
    file.writelines(L10)
    file.writelines(L11)
    file.writelines(L12)
    file.close()
if  valz==0 and val == 1:
    file = open("Finalne_slike/podacislike{}.txt".format(broj_slike), "w")
    L1 = ["Da li ima zbunjastih poljana : NE \n"]
    L2 = ["Broj superpixela: {} \n".format(len(np.unique(segments)))]
    L3 = ["Preciznost:{} \n".format((len(nova_lista) / val1) * 100)]
    L4 = ["Vrijeme izvršavanja koda: {} \n".format(ukupno_vrijeme)]
    L5 = ["Tip slike: {} \n".format(tip_slike)]
    file.write("Podaci slike {}: \n".format(broj_slike))
    file.writelines(L1)
    file.writelines(L2)
    file.writelines(L3)
    file.writelines(L4)
    file.writelines(L5)
if  valz==1 and val == 1:
    file = open("Finalne_slike/podacislike{}.txt".format(broj_slike), "w")
    L1 = ["Da li ima zbunjastih poljana : DA \n"]
    L2 = ["Parametri prve morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem, tipmorfo)]
    L3 = ["Parametri drugog bluranja : Diameter: {} , Color: {} , Space : {} , Kernel: {} , Tip bluranja: {} \n".format(diameter_min1, color_min1, space_min1, ksizeb1, blur_type1)]
    L4 = ["Parametri drugog odredjivanja ivica : Min.treshold:{},Max.treshold:{},Aperture:{},Gradient:{},Blocksize:{},Sigma:{},a1:{},Edge metoda: {} \n".format(mintres1, maxtres1, aperture1, gradient1, blocksize1, sigma1, a1, edge_type1)]
    L5 = ["Parametri druge morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem1, tipmorfo2)]
    L6 = ["Parametri treće morfologije : Ksize: {}, Tip_morfologije: {} \n".format(ksizem2, tipmorfo3)]
    L7 = ["Broj superpixela: {} \n".format(len(np.unique(segments)))]
    L8 = ["Preciznost:{} \n".format((len(nova_lista) / val1) * 100)]
    L9 = ["Vrijeme izvršavanja koda: {} \n".format(ukupno_vrijeme)]
    L10 = ["Tip slike: {} \n".format(tip_slike)]
    file.write("Podaci slike {}: \n".format(broj_slike))
    file.writelines(L1)
    file.writelines(L2)
    file.writelines(L3)
    file.writelines(L4)
    file.writelines(L5)
    file.writelines(L6)
    file.writelines(L7)
    file.writelines(L8)
    file.writelines(L9)
    file.writelines(L10)
