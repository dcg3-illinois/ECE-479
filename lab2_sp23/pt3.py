import cv2
import picamera
import numpy as np
from mtcnn.mtcnn import MTCNN

def capture_image():
    # Instrctor note: this can be directly taken from the PiCamera documentation
    # Create the in-memory stream
    stream = io.BytesIO()
    with picamera.PiCamera() as camera:
        camera.capture(stream, format='jpeg')
        
    # Construct a numpy array from the stream
    data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
    
    # "Decode" the image from the array, preserving colour
    image = cv2.imdecode(data, 1)
    
    # OpenCV returns an array with data in BGR order. 
    # The following code invert the order of the last dimension.
    image = image[:, :, ::-1]
    return image


def detect_and_crop(mtcnn, image):
    detection = mtcnn.detect_faces(image)[0]
    #TODO
    #extract the bounding box
    x, y, width, height = detection['box']
    x = int(x*0.9)
    y = int(y*0.9)
    width = int(width*1.2)
    height = int(height*1.2)

    cropped_image = image[y:y+height, x:x+width]
    
    return cropped_image, (x,y,width,height)

# function provided for the students to draw the rectangle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
def show_bounding_box(image, bounding_box):
    x1, y1, w, h = bounding_box
    fig, ax = plt.subplots(1,1)
    ax.imshow(image)
    ax.add_patch(Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()
    return

def pre_process(face, required_size=(160, 160)):
    ret = cv2.resize(face, required_size)
    ret = ret.astype('float32')
    mean, std = ret.mean(), ret.std()
    ret = (ret - mean) / std
    return ret

def run_model(model, face):
# students will need to fill in the following function
    #TODO
    # Get input and output tensors.
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    model.set_tensor(input_details[0]['index'], face.reshape(input_shape))
    
    model.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = model.get_tensor(output_details[0]['index'])
    
    return output_data

def read_image(file):
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# mtcnn = MTCNN()
# image = capture_image()
# cropped_image, dim = detect_and_crop(mtcnn, image)

# tfl_file = "inception_lite.tflite" #change this to the inception model
# interpreter = tf.lite.Interpreter(model_path=tfl_file)
# interpreter.allocate_tensors()
# #preprocess the face
# face = pre_process(cropped_image)
# #run the model with the above funtion
# output_data = run_model(interpreter, face)

# process the second image of the first person

# 1. Read the image
mtcnn = MTCNN()
#image = capture_image()
image = read_image("reynolds.jpg")
# 2. Detect and Crop
cropped_image, dim = detect_and_crop(mtcnn, image)
# 3. Proprocess
tfl_file = "./inception_lite"
interpreter = tf.lite.Interpreter(model_path=tfl_file)
interpreter.allocate_tensors()
#preprocess the face
face = pre_process(cropped_image)
# 4. Run the model
output_data = run_model(interpreter, face)

# process the image of the second person
image2 = read_image("reynolds2.jpg")
cropped_image2, dim2 = detect_and_crop(mtcnn, image2)
#preprocess the face
face2 = pre_process(cropped_image2)
# 4. Run the model
output_data2 = run_model(interpreter, face2)


# Do the comparison of the distance
# print(output_data[0])
# print(output_data2[0])
print(np.linalg.norm(output_data[0] - output_data2[0]))