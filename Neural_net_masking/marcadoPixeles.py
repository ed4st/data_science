# importing the module
import sys 
import pandas as pd
import cv2

# Lista de coordenadas de pixeles del objeto marcados
fglist = list()
# Lista de coordenadas de pixeles del fondo marcados
bglist = list()
# Indicador de si se debe registrar las coordenadas del mouse cuando se mueva
bMark  = False
# Indicador de si las coordenadas son de pixeles del fondo o del objeto
bBG    = True

def click_event(event, x, y, flags, params):
    global bMark
    global bBG
    # Al presionar el boton izquierdo del mouse se inicia el registro de coordenadas
    if event==cv2.EVENT_LBUTTONDOWN:
        bMark = True
    # Al liberar el boton izquierdo del mouse se deja de registrar las coordenadas
    if event==cv2.EVENT_LBUTTONUP:
        bMark = False
    # Si el mouse se mueve y está presionado el boton izquierdo, se registran las
    # coordenadas del mouse en lista de pixeles del fondo o del objeto
    if event==cv2.EVENT_MOUSEMOVE and bMark:
        if bBG:
            bglist.append((x,y))
            cv2.circle(img,(x,y),3,(255,0,0),-1)
        else:
            fglist.append((x,y))
            cv2.circle(img,(x,y),3,(0,0,255),-1)


# driver function
if __name__=="__main__":
    if len(sys.argv)<4:
        print('Forma de uso:')
        print('    python marcadoPixeles.py imagen_entrada imagen_salida CSV')
        print('donde')
        print('    imagen_entrada - Nombre de la imagen de entrada')
        print('    imagen_salida  - Nombre de la imagen con los pixeles marcados')
        print('    CSV    - Nombre del archivo CSV  de salida\n')
        sys.exit(0)

    print('\nIndicaciones:')
    print('- Presione el botón izquierdo del mouse para empezar a marcar los pixeles ')
    print('  del fondo de la imagen cuando desplaza el mouse, mantenido presiondo el botón.')
    print('- Libere el botón izquierdo del mouse para dejar de marcar los pixeles')
    print('- Presione la tecla "o" para marcar los pixeles del objeto.')
    print('- Presione la tecla "f" para marcar los pixeles del fondo.')
    print('- Presione la tecla ESC terminar y grabar el archivo CSV de salida y la imagen marcada')

    input_img  = sys.argv[1]
    output_img = sys.argv[2]
    file_csv   = sys.argv[3]
 
    # reading the image
    img = cv2.imread(input_img)
    orig_img = img.copy()
    hsv_orig_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
 
    cv2.namedWindow("Imagen")
 
    # setting mouse hadler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('Imagen', click_event)
 
    # wait for a key to be pressed to exit
    while True:
        cv2.imshow('Imagen', img)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('o'):
            bBG = False
        if key==ord('f'):
            bBG = True
        if key==27 or key==ord('c'):
            break 

    # close the window
    cv2.destroyAllWindows()

    cv2.imwrite(output_img, img)
    
    xaux, yaux = fglist[0]
    print(orig_img[xaux,yaux])
    print(hsv_orig_img[xaux,yaux])
    print('Número pixeles en el fondo:', len(bglist))
    print('Número pixeles en el objeto:', len(fglist))

    # Se graban los colores de los pixeles marcados en un archivo CSV
    file1 = open(file_csv,  "w")   
    for x,y in bglist:
        b = orig_img[y, x, 0]
        g = orig_img[y, x, 1]
        r = orig_img[y, x, 2]

        h = hsv_orig_img[y, x, 0]
        s = hsv_orig_img[y, x, 1]
        v = hsv_orig_img[y, x, 2]
        file1.write("%d,%d,%d,%d,%d,%d,%d\n" % (b,g,r,h,s,v,0))
    for x,y in fglist:
        b = orig_img[y, x, 0]
        g = orig_img[y, x, 1]
        r = orig_img[y, x, 2]
        h = hsv_orig_img[y, x, 0]
        s = hsv_orig_img[y, x, 1]
        v = hsv_orig_img[y, x, 2]
        file1.write("%d,%d,%d,%d,%d,%d,%d\n" % (b,g,r,h,s,v,1))

file1.close() 
        

