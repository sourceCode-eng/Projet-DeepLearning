from keras_cv.models import StableDiffusion
model = StableDiffusion (img_width=512,img_height=512)
images = model.text_to_image(
    "photograph of a gamer",
    "on the surface of the moon riding a car",
    batch_size=2
)