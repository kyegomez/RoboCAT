from robocat.model import ImageDataGenerator

model = ImageDataGenerator()
img = model.generate(prompt="A robot looking at a soda can in first perrson")
print(img)