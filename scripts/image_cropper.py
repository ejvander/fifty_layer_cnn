from PIL import Image
import sys

def crop_img(img_in, img_seg):
  pad_size = 10 # Tweak this if we need to

  o_img = load_image(img_in)
  seg_img = load_image(img_seg)

  width  = o_img.width
  height = o_img.height

  boundary = find_img_boundary(seg_img)
  # Boundary is an exact boudary, so we'll pad it a little to give a little breathing room
  boundary = pad_boundary(boundary, seg_img.width, seg_img.height, pad_size)
  boundary = square_boundary(boundary, seg_img.width, seg_img.height)

  cropped_img = o_img.crop((boundary[2], boundary[0], boundary[3], boundary[1]))

  return cropped_img

def load_image(img):
  if(type(img) is str):
    # Load image from string
    return Image.open(img)  # loading image
  else: 
    # Else, assume it's already an image object
    return img


# Returns list boundary (top, bottom, left, right)
def find_img_boundary(img):
  top = img.height
  bottom = 0
  left = img.width
  right = 0

  img_pxls = list(img.getdata())

  found_pixel = 0
  for index, x in enumerate(img_pxls):
    #background pixels are black(0,0,0,255)
    if x[0] != 0:
      found_pixel = 1
      if((index//img.width) < top):
        top = (index//img.width)
      if((index//img.width) > bottom):
        bottom = (index//img.width)
      if((index%img.width) < left):
        left = (index%img.width)
      if((index%img.width) > right):
        right = (index%img.width)
  
  # In case we can't find any segmentation mask, keep the origional size
  if(found_pixel == 0):
    top = 0
    bottom = img.height
    left = 0
    right = img.width

  return [top, bottom, left, right]

# Pad boundary by pad amount, max out at width, height
def pad_boundary(boundary, width, height, pad):
  new_boundary = boundary
  # Pad top
  new_boundary[0] = 0 if (boundary[0] - pad < 0) else boundary[0] - pad
  # Pad bottom
  new_boundary[1] = height if (boundary[1] + pad > height) else boundary[1] + pad
  # Pad left
  new_boundary[2] = 0 if (boundary[2] - pad < 0) else boundary[2] - pad
  # Pad bottom
  new_boundary[3] = width if (boundary[3] + pad > width) else boundary[3] + pad

  return new_boundary

# Make the boundary a square
def square_boundary(boundary, width, height):
  new_boundary = boundary
  bound_height = boundary[1] - boundary[0]
  bound_width = boundary[3] - boundary[2]

  diff = abs(bound_width - bound_height)
  diffdiv2 = diff//2
  diffodd = diff%2
  if(bound_width > bound_height):
    # Increase height
    # If the padded version won't be too big, just pad each side
    # If it's an odd number for the diff, just add to the top
    if((boundary[0] - diffdiv2 - diffodd >= 0) and (boundary[1] + diffdiv2 < height)):
      new_boundary[0] = boundary[0] - diffdiv2 - diffodd
      new_boundary[1] = boundary[1] + diffdiv2
    elif((boundary[0] - diffdiv2 - diffodd < 0)):
      new_boundary[0] = 0
      new_boundary[1] = bound_width
    else:
      new_boundary[0] = height - bound_width
      new_boundary[1] = height
  elif(bound_height > bound_width):
    # Increase width
    # If the padded version won't be too big, just pad each side
    # If it's an odd number for the diff, just add to the left
    if((boundary[2] - diffdiv2 - diffodd >= 0) and (boundary[3] + diffdiv2 < width)):
      new_boundary[2] = boundary[2] - diffdiv2 - diffodd
      new_boundary[3] = boundary[3] + diffdiv2
    elif((boundary[2] - diffdiv2 - diffodd < 0)):
      new_boundary[2] = 0
      new_boundary[3] = bound_height
    else:
      new_boundary[2] = width - bound_height
      new_boundary[3] = width
   
  return new_boundary




if __name__ == "__main__":
  input_img = sys.argv[1]
  segment_img = sys.argv[2]
  output_img = sys.argv[3]

  cropped_img = crop_img(input_img, segment_img)
  
  cropped_img.save(output_img)

