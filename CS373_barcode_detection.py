# Built in packages
import math
import sys
from pathlib import Path

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# import our basic, light-weight png reader library
import imageIO.png

# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# a useful shortcut method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array

# returns colour array from rgb arrays
def separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height):
    new_array = [[[0 for c in range(3)] for x in range(image_width)] for y in range(image_height)]

    for y in range(image_height):
        for x in range(image_width):
            new_array[y][x][0] = px_array_r[y][x]
            new_array[y][x][1] = px_array_g[y][x]
            new_array[y][x][2] = px_array_b[y][x]

    return new_array

# All of the functions below are written by me (jwei578) and only me in coderunner, except for the queue class implementation which is borrowed from a code runner question

# Converts rgb pixel arrays to one greyscale array
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            greyscale_pixel_array[i][j] = round(0.299 * pixel_array_r[i][j] + 0.587 * pixel_array_g[i][j] + 0.114 * pixel_array_b[i][j])
    return greyscale_pixel_array

# Returns min,max values given a px array
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    min = 256
    max = -1
    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] < min):
                min = pixel_array[i][j]
            if (pixel_array[i][j] > max):
                max = pixel_array[i][j]
    return min, max

# Normalize px array to stretch between 0 to 255
def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    (min, max) = computeMinAndMaxValues(pixel_array, image_width, image_height)
    if (min == max):
        return greyscale_pixel_array

    for i in range(image_height):
        for j in range(image_width):
            # normalize
            greyscale_pixel_array[i][j] = round(255*(pixel_array[i][j] - min)/(max-min))
    return greyscale_pixel_array

# std px array pixels with a 5x5 neighbourhood size
def computeStandardDeviationImage5x5(pixel_array, image_width, image_height):
    returnArray = createInitializedGreyscalePixelArray(image_width, image_height)
    medArray = []
    stdArray = []
    
    for i in range(1,image_height-1):
        for j in range(1,image_width-1):
            medArray = []
            stdArray = []
            for m in range(i-2, i+3):
                for n in range(j-2, j+3):
                    if(m < 0 or m > image_height-1 or n < 0 or n > image_width-1):
                        continue
                    else:
                        medArray.append(pixel_array[m][n])
            avg = float(sum(medArray))/len(medArray)
            for item in medArray:
                stdArray.append(math.pow(item-avg, 2))
            returnArray[i][j] = math.sqrt(sum(stdArray)/9)
    
    return returnArray

# Returns 3x3 Gaussian filtered px array, with repeated at border handling. Horrible horrible code
def computeGaussianAveraging3x3RepeatBorder(pixel_array, image_width, image_height):
    returnArray = createInitializedGreyscalePixelArray(image_width, image_height)
    
    if (image_height == 1):
        return pixel_array


    for i in range(image_height):
        for j in range(image_width):
            if (i == 0 and j == 0):
                a11 = pixel_array[i][j]
                a12 = pixel_array[i][j]
                a13 = pixel_array[i][j+1]
                a21 = pixel_array[i][j]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i+1][j]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j+1]
            elif (i == 0 and j == image_width-1):
                a11 = pixel_array[i][j-1]
                a12 = pixel_array[i][j]
                a13 = pixel_array[i][j]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j]
                a31 = pixel_array[i+1][j-1]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j]
            elif (i == image_height-1 and j == 0):
                a11 = pixel_array[i-1][j]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j+1]
                a21 = pixel_array[i][j]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i][j]
                a32 = pixel_array[i][j]
                a33 = pixel_array[i][j+1]
            elif (i == image_height-1 and j == image_width-1):
                a11 = pixel_array[i-1][j-1]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j]
                a31 = pixel_array[i][j-1]
                a32 = pixel_array[i][j]
                a33 = pixel_array[i][j]
            elif (i == 0 and 0 < j < image_width-1):
                a11 = pixel_array[i][j-1]
                a12 = pixel_array[i][j]
                a13 = pixel_array[i][j+1]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i+1][j-1]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j+1]
            elif (i == image_height-1 and 0 < j < image_width-1):
                a11 = pixel_array[i-1][j-1]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j+1]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i][j-1]
                a32 = pixel_array[i][j]
                a33 = pixel_array[i][j+1]
            elif (0 < i < image_height-1 and j == 0):
                a11 = pixel_array[i-1][j]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j+1]
                a21 = pixel_array[i][j]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i+1][j]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j+1]
            elif (0 < i < image_height-1 and j == image_width-1):
                a11 = pixel_array[i-1][j-1]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j]
                a31 = pixel_array[i+1][j-1]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j]
            else:
                a11 = pixel_array[i-1][j-1]
                a12 = pixel_array[i-1][j]
                a13 = pixel_array[i-1][j+1]
                a21 = pixel_array[i][j-1]
                a22 = pixel_array[i][j]
                a23 = pixel_array[i][j+1]
                a31 = pixel_array[i+1][j-1]
                a32 = pixel_array[i+1][j]
                a33 = pixel_array[i+1][j+1]
                
            returnArray[i][j] = (a11 + 2*a12 + a13 + 2*a21 + 4*a22 + 2*a23 + a31 + 2*a32 + a33)/16
    
    return returnArray

# Return a threshold px array given the threshold
def computeThreshold(pixel_array, image_width, image_height, threshold):
    returnArr = createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0)

    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] >= threshold):
                returnArr[i][j] = 255
    return returnArr

# Dilate the px array with 5x5 filter
def computeDilation8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    outputArr = createInitializedGreyscalePixelArray(image_width, image_height)
    SEList = []
    
    for i in range(image_height):
        for j in range(image_width):
            SEList = []
            for m in range(i-2, i+3):
                for n in range (j-2, j+3):
                    if(m < 0 or m > image_height-1 or n < 0 or n > image_width-1):
                        SEList.append(0)
                    else:
                        SEList.append(pixel_array[m][n])
            for pixel in SEList:
                if (pixel != 0):
                    outputArr[i][j] = 1
    return outputArr

# Erode the px array with 5x5 filter
def computeErosion8Nbh5x5FlatSE(pixel_array, image_width, image_height):
    outputArr = createInitializedGreyscalePixelArray(image_width, image_height)
    SEList = []
    
    for i in range(image_height):
        for j in range(image_width):
            SEList = []
            for m in range(i-2, i+3):
                for n in range (j-2, j+3):
                    if(m < 0 or m > image_height-1 or n < 0 or n > image_width-1):
                        SEList.append(0)
                    else:
                        SEList.append(pixel_array[m][n])
            if (0 not in SEList):
                outputArr[i][j] = 1
    return outputArr

# A class implementing queue data structure
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        self.items.insert(0,item)

    def dequeue(self):
        return self.items.pop()

    def size(self):
        return len(self.items)

# Returns a pixel array with each component colored by its ID, and a dictionary of ID corresponding to size of component
def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    labelID = 0
    visitedPixel = createInitializedGreyscalePixelArray(image_width, image_height)
    dictionary = {}
    resultPixel = createInitializedGreyscalePixelArray(image_width, image_height)
    
    for i in range(image_height):
        for j in range(image_width):
            # found unvisited pixel
            if (visitedPixel[i][j] == 0):
                visitedPixel[i][j] = 1
                # found new component
                if (pixel_array[i][j] != 0):
                    labelID += 1
                    pixelCnt = 1
                    componentQueue = Queue()
                    componentQueue.enqueue((i,j))
                    resultPixel[i][j] = labelID
                    # search and record size and component ID
                    while (not componentQueue.isEmpty()):
                        # search surronding surrounding connected pixels
                        x, y = componentQueue.dequeue()
                        for m in range(x-1, x+2):
                            for n in range(y-1, y+2):
                                if (m < 0 or m > image_height-1 or n < 0 or n > image_width-1):
                                    continue
                                elif (m != x and n !=y):
                                    continue
                                elif (visitedPixel[m][n] == 0 and pixel_array[m][n] != 0):
                                    # found connected
                                    componentQueue.enqueue((m,n))
                                    resultPixel[m][n] = labelID
                                    pixelCnt += 1
                                    visitedPixel[m][n] = 1
                                else:
                                    # not connected
                                    visitedPixel[m][n] = 1
                                    continue
                    # record ID and size at the end            
                    dictionary[labelID] = pixelCnt
    return resultPixel, dictionary

# Find the min, max, x and y values of the component based on its ID
def findMinMaxComponentCoordinates(pixel_array, image_width, image_height, ID):
    minX = 10000
    minY = 10000
    maxX = 0
    maxY = 0
    for i in range(image_height):
        for j in range(image_width):
            if (pixel_array[i][j] == ID):
                if (i > maxY):
                    maxY = i
                if (j > maxX):
                    maxX = j
                if (i < minY):
                    minY = i
                if (j < minX):
                    minX = j
    return minX, minY, maxX, maxY

# This is our code skeleton that performs the barcode detection.
# Feel free to try it on your own images of barcodes, but keep in mind that with our algorithm developed in this assignment,
# we won't detect arbitrary or difficult to detect barcodes!
def main():

    command_line_arguments = sys.argv[1:]

    SHOW_DEBUG_FIGURES = True

    # this is the default input image filename
    filename = "Multiple_barcodes"
    input_filename = "images/"+filename+".png"

    if command_line_arguments != []:
        input_filename = command_line_arguments[0]
        SHOW_DEBUG_FIGURES = False

    output_path = Path("output_images")
    if not output_path.exists():
        # create output directory
        output_path.mkdir(parents=True, exist_ok=True)

    output_filename = output_path / Path(filename+"_output.png")
    if len(command_line_arguments) == 2:
        output_filename = Path(command_line_arguments[1])

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)

    # setup the plots for intermediate results in a figure
    fig1, axs1 = pyplot.subplots(2, 2)
    axs1[0, 0].set_title('Input red channel of image')
    axs1[0, 0].imshow(px_array_r, cmap='gray')
    axs1[0, 1].set_title('Input green channel of image')
    axs1[0, 1].imshow(px_array_g, cmap='gray')
    axs1[1, 0].set_title('Input blue channel of image')
    axs1[1, 0].imshow(px_array_b, cmap='gray')

    # STUDENT IMPLEMENTATION here
    # Convert to greyscale
    greyScaleArr = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    # Normalise, stretch the values between 0 and 255
    normGreyScaleArr = scaleTo0And255AndQuantize(greyScaleArr, image_width, image_height)

    # Standard deviation method, Apply a 5x5 standard deviation filter
    contrastArr = computeStandardDeviationImage5x5(normGreyScaleArr, image_width, image_height)

    # Gaussian filter, 3x3 gaussian filter, try repeating the filter four times or a larger sigma and filter size
    blurArr = computeGaussianAveraging3x3RepeatBorder(contrastArr, image_width, image_height)
    for i in range(3):
        blurArr = computeGaussianAveraging3x3RepeatBorder(blurArr, image_width, image_height)

    # Threshhold the image, good threshhold for the standard deviation method is around 25, for gradient method, around 100
    bnwThresholdArr = computeThreshold(blurArr, image_width, image_height, 25)

    # Erosion and dilation, for standard deviation method, 2 or three consecutive erosion steps followed by two consecutive dilation steps with a 5x5 filter. For the image gradient method, two consecutive dilation 
    # followed by two erosions may work better.
    openArr = computeErosion8Nbh5x5FlatSE(bnwThresholdArr, image_width, image_height)
    openArr = computeErosion8Nbh5x5FlatSE(openArr, image_width, image_height)
    openArr = computeErosion8Nbh5x5FlatSE(openArr, image_width, image_height)
    openArr = computeDilation8Nbh5x5FlatSE(openArr, image_width, image_height)
    openArr = computeDilation8Nbh5x5FlatSE(openArr, image_width, image_height)

    # Connected component analysis to find the largest connected component
    (componentGraph, dictionary) = computeConnectedComponentLabeling(openArr, image_width, image_height)
    largestComponentID = max(dictionary, key = dictionary.get)

    # Compute a bounding box
    # Change these values based on the detected barcode region from your algorithm
    (bbox_min_x, bbox_min_y, bbox_max_x, bbox_max_y) = findMinMaxComponentCoordinates(componentGraph, image_width, image_height, largestComponentID)
    
    px_array = separateArraysToRGB(px_array_r, px_array_g, px_array_b, image_width, image_height)
    # px_array = blurArr

    # The following code is used to plot the bounding box and generate an output for marking
    # Draw a bounding box as a rectangle into the input image
    axs1[1, 1].set_title('Final image of detection')
    axs1[1, 1].imshow(px_array, cmap='gray')
    rect = Rectangle((bbox_min_x, bbox_min_y), bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y, linewidth=2,
                    edgecolor='g', facecolor='none')
    axs1[1, 1].add_patch(rect)

    # write the output image into output_filename, using the matplotlib savefig method
    extent = axs1[1, 1].get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    pyplot.savefig(output_filename, bbox_inches=extent, dpi=600)

    if SHOW_DEBUG_FIGURES:
        # plot the current figure
        pyplot.show()


if __name__ == "__main__":
    main()