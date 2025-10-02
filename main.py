import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./input/logo_rhine.png', cv2.IMREAD_UNCHANGED)

sizeX = img.shape[1]
sizeY = img.shape[0]

weight1 = 1
weight2 = 0.5

light_len = 0.97

SCATTERING_POW = 4.0
SCATTERING_STUCK = 0.001

# Convert the image to RGBA if it is not already
if img.shape[2] != 4:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    print("Converted image to RGBA format.")

# Extract the Alpha channel
alpha_channel = img[:, :, 3]

# Apply the Sobel filter in the vertical direction
sobel_vertical = cv2.Sobel(alpha_channel, cv2.CV_64F, 0, 1, ksize=5, scale=1)

sobel_1 = np.where(sobel_vertical > 0, sobel_vertical, 0)
sobel_1 = cv2.normalize(sobel_1, None, 0, 255, cv2.NORM_MINMAX)

sobel_2 = np.abs(sobel_vertical)
sobel_2 = cv2.normalize(sobel_2, None, 0, 255, cv2.NORM_MINMAX)

# Create a color version of the Sobel result
sobel_color1 = cv2.cvtColor(sobel_1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
sobel_color2 = cv2.cvtColor(sobel_2.astype(np.uint8), cv2.COLOR_GRAY2BGR)

scattering_light = np.full((sizeY, sizeX), 0.0, dtype=np.float64)

for x in range(sizeX):
    scattering_light[0, x] = 1.0 * (1.0-SCATTERING_STUCK*pow(img[0,x,3]/255.0, SCATTERING_POW))

for y in range(sizeY-1):
    for x in range(sizeX):
        scattering_light[y, x] = scattering_light[y, x]*light_len + (1.0-light_len)*(1.0-img[y,x,3]/255.0)
        if x == 0:
            scattering_light[y+1, x] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x,3]/255.0, SCATTERING_POW))
            scattering_light[y+1, x+1] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x+1,3]/255.0, SCATTERING_POW))
        elif x == sizeX-1:
            scattering_light[y+1, x] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x,3]/255.0, SCATTERING_POW))
            scattering_light[y+1, x-1] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x-1,3]/255.0, SCATTERING_POW))
        else:
            #scattering_light[y+1, x] += scattering_light[y, x]*0.34 * (1.0-SCATTERING_STUCK*pow(img[y+1,x,3]/255.0, 2))
            scattering_light[y+1, x-1] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x-1,3]/255.0, SCATTERING_POW))
            scattering_light[y+1, x+1] += scattering_light[y, x]*0.5 * (1.0-SCATTERING_STUCK*pow(img[y+1,x+1,3]/255.0, SCATTERING_POW))

scattering_light = np.clip(scattering_light*255.0, 0, 255).astype(np.uint8)

#scattering_light = cv2.equalizeHist(scattering_light)

# Combine the original image with the Sobel result, preserving the alpha channel
#result = cv2.addWeighted(img[:, :, :3], 1, sobel_color1, weight1, 0)
#result = cv2.addWeighted(result, 1, sobel_color2, weight2, 0)

# Add the alpha channel back to the result (correcting the channel order)
#result = cv2.merge((result[:, :, 0], result[:, :, 1], result[:, :, 2], alpha_channel))

# Save the result as a PNG file
cv2.imwrite('./output/HL1.png', sobel_1)
cv2.imwrite('./output/HL2.png', sobel_2)
cv2.imwrite('./output/SCT.png', scattering_light)
# Display the result
plt.imshow(scattering_light, cmap='gray')
plt.title('IC02: Scattering Light')
plt.axis('off')
plt.show()