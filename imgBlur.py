from skimage import io as skio
from skimage import filters
from skimage.color import rgb2gray
from skimage.feature import canny
from scipy import ndimage as ndi
from skimage import morphology
from skimage import segmentation
from PIL import Image

def imgBlurFunc(csv_dir, img_dir):
    csvfile = open(csv_dir)
    reader = csv.reader(csvfile)

    # 讀取csv標籤
    labels = []
    for line in reader:
        tmp = [line[0],line[1]]
        # print tmp
        labels.append(tmp)
    csvfile.close()
    
    # remove the first row
    labels = labels[1:]
    
    # 轉換圖片的標籤
    for i in range(len(labels)):
        labels[i][1] = labels[i][1].replace("A","0")
        labels[i][1] = labels[i][1].replace("B","1")
        labels[i][1] = labels[i][1].replace("C","2")
    
    
    for i in range(len(items)):
        #img = cv2.imread("C1-P1_Train/" + labels[i][0] )
        #res = cv2.resize(img,(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        #res = img_to_array(res)
        #remove/blur bg
        img = skio.imread("C1-P1_Train/" + labels[i][0] )
        gray_img0 = img[:,:,0]-img[:,:,1]
        gray_img1 = img[:,:,0]-img[:,:,2]
        gray_img2 = rgb2gray(img)
        sobel0 = filters.sobel(gray_img0)
        sobel1 = filters.sobel(gray_img1)
        sobel2 = filters.sobel(gray_img2)
        sobel = (sobel0+sobel2+sobel1)
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.rcParams['figure.dpi'] = 200
        blurred = filters.gaussian(sobel, sigma=1)
        light_spots = np.array((img> 245).nonzero()).T
        dark_spots = np.array((img[:,:,0] < 20).nonzero()).T
        bool_mask = np.zeros(img.shape, dtype=np.bool)
        bool_mask[tuple(light_spots.T)] = False
        bool_mask[tuple(dark_spots.T)] = True
        bound1=img.shape[0]/2-168
        bound2=img.shape[0]/2+168
        bound3=img.shape[1]/2-168
        bound4=img.shape[1]/2+168
        bool_mask[int(bound1):int(bound2),int(bound3):int(bound4),0:2]=False
        seed_mask, num_seeds = ndi.label(bool_mask)
        ws0 = segmentation.watershed(blurred, seed_mask[:,:,0])
        ws1 = segmentation.watershed(blurred, seed_mask[:,:,0])
        ws2 = segmentation.watershed(blurred, seed_mask[:,:,0])
        ws = (ws0+ws1+ws2)//3
        background = max(set(ws.ravel()), key=lambda g: np.sum(ws == g))
        background_mask = (ws == background)
        img_blur=np.copy(img)
        img_blur= filters.gaussian(img_blur, sigma=20, multichannel=True)
        img_cut=np.copy(img)
        img_cut[:,:,0] = img[:,:,0]-img[:,:,0] * ~background_mask
        img_cut[:,:,1] = img[:,:,1]-img[:,:,1] * ~background_mask
        img_cut[:,:,2] = img[:,:,2]-img[:,:,2] * ~background_mask
        background=Image.fromarray(np.uint8(img_blur*255))
        fg=Image.fromarray(np.uint8(img_cut))
        background.paste(fg,(0,0),Image.fromarray(((background_mask)*255).astype(np.uint8)))

        open_cv_image = np.array(background)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        res = cv2.resize(open_cv_image,(image_size,image_size),interpolation=cv2.INTER_LINEAR)
        res = img_to_array(res)
        X.append(res)    
        y.append(labels[i][1])