from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes


class LocalCV:
    def __init__(self, computervision_client, path):
        self.computervision_client = computervision_client
        self.local_image_path = path

    def get_image_path(self):
        return self.local_image_path

    def set_image_path(self, path):
        self.local_image_path = path

    def show_image_path(self):
        print(self.local_image_path)

    def describe_image(self):

        print("===== Describe an Image - local =====")
        # Open local image file
        local_image = open(self.local_image_path, "rb")

        # Call API
        description_result = self.computervision_client.describe_image_in_stream(local_image)

        # Get the captions (descriptions) from the response, with confidence level
        print("Description of local image: ")
        if (len(description_result.captions) == 0):
            print("No description detected.")
        else:
            for caption in description_result.captions:
                print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))
        print()

    def get_describe_image(self):
        tmp_str = ""

        tmp_str += "===== Describe an Image - local =====\n"
        # Open local image file
        local_image = open(self.local_image_path, "rb")

        # Call API
        description_result = self.computervision_client.describe_image_in_stream(local_image)

        # Get the captions (descriptions) from the response, with confidence level
        tmp_str += "Description of local image: \n"
        if (len(description_result.captions) == 0):
            tmp_str += "No description detected.\n"
        else:
            for caption in description_result.captions:
                tmp_str += "'{}' with confidence {:.2f}%\n".format(caption.text, caption.confidence * 100)
        return tmp_str
        

    def categorize_image(self):
        print("===== Categorize an Image - local =====")
        # Open local image file
        local_image = open(self.local_image_path, "rb")
        # Select visual feature type(s)
        local_image_features = ["categories"]
        # Call API
        categorize_results_local = self.computervision_client.analyze_image_in_stream(local_image, local_image_features)

        # Print category results with confidence score
        print("Categories from local image: ")
        if (len(categorize_results_local.categories) == 0):
            print("No categories detected.")
        else:
            for category in categorize_results_local.categories:
                print("'{}' with confidence {:.2f}%".format(category.name, category.score * 100))
        print()

    def get_categorize_image(self):
        tmp_str = ""
        # tmp_str += "===== Categorize an Image - local =====\n"
        # Open local image file
        local_image = open(self.local_image_path, "rb")
        # Select visual feature type(s)
        local_image_features = ["categories"]
        # Call API
        categorize_results_local = self.computervision_client.analyze_image_in_stream(local_image, local_image_features)

        # Print category results with confidence score
        # tmp_str += "Categories from local image: \n"
        if (len(categorize_results_local.categories) == 0):
            tmp_str += "No categories detected.\n"
        else:
            for category in categorize_results_local.categories:
                if (category.score * 100) > 60:
                    tmp_str += "'{}', ".format(category.name)
        tmp_str += "\n"
        return tmp_str

    def tag_image(self):
        print("===== Tag an Image - local =====")
        # Open local image file
        local_image = open(self.local_image_path, "rb")
        # Call API local image
        tags_result_local = self.computervision_client.tag_image_in_stream(local_image)

        # Print results with confidence score
        print("Tags in the local image: ")
        if (len(tags_result_local.tags) == 0):
            print("No tags detected.")
        else:
            for tag in tags_result_local.tags:
                print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))
        print()

    def get_tag_image(self):
        tmp_str = ""
        # Open local image file
        local_image = open(self.local_image_path, "rb")
        # Call API local image
        tags_result_local = self.computervision_client.tag_image_in_stream(local_image)

        # Print results with confidence score
        # tmp_str += "Tags in the local image: \n"
        if (len(tags_result_local.tags) == 0):
            tmp_str += "No tags detected.\n"
        else:
            for tag in tags_result_local.tags:
                if (tag.confidence * 100) > 60:
                    tmp_str += "'{}', ".format(tag.name)
        tmp_str += "\n"
        return tmp_str

    def detect_face(self):
        print("===== Detect Faces - local =====")
        # Open local image
        local_image = open(self.local_image_path, "rb")
        # Select visual features(s) you want
        local_image_features = ["faces"]
        # Call API with local image and features
        detect_faces_results_local = self.computervision_client.analyze_image_in_stream(local_image,
                                                                                        local_image_features)

        # Print results with confidence score
        print("Faces in the local image: ")
        if (len(detect_faces_results_local.faces) == 0):
            print("No faces detected.")
        else:
            for face in detect_faces_results_local.faces:
                print("'{}' of age {} at location {}, {}, {}, {}".format(face.gender, face.age, \
                                                                         face.face_rectangle.left,
                                                                         face.face_rectangle.top, \
                                                                         face.face_rectangle.left + face.face_rectangle.width, \
                                                                         face.face_rectangle.top + face.face_rectangle.height))
        print()

    '''
    Detect Adult or Racy Content - local
    This example detects adult or racy content in a local image, then prints the adult/racy score.
    The score is ranged 0.0 - 1.0 with smaller numbers indicating negative results.
    '''

    def detect_adult_or_racy(self):
        print("===== Detect Adult or Racy Content - local =====")
        # Open local file
        local_image = open(self.local_image_path, "rb")
        # Select visual features you want
        local_image_features = ["adult"]
        # Call API with local image and features
        detect_adult_results_local = self.computervision_client.analyze_image_in_stream(local_image,
                                                                                        local_image_features)

        # Print results with adult/racy score
        print("Analyzing local image for adult or racy content ... ")
        print("Is adult content: {} with confidence {:.2f}".format(detect_adult_results_local.adult.is_adult_content,
                                                                   detect_adult_results_local.adult.adult_score * 100))
        print("Has racy content: {} with confidence {:.2f}".format(detect_adult_results_local.adult.is_racy_content,
                                                                   detect_adult_results_local.adult.racy_score * 100))
        print()

    '''
    Detect Color - local
    This example detects the different aspects of its color scheme in a local image.
    '''

    def detect_color(self):
        print("===== Detect Color - local =====")
        # Open local image
        local_image = open(self.local_image_path, "rb")
        # Select visual feature(s) you want
        local_image_features = ["color"]
        # Call API with local image and features
        detect_color_results_local = self.computervision_client.analyze_image_in_stream(local_image,
                                                                                        local_image_features)

        # Print results of the color scheme detected
        print("Getting color scheme of the local image: ")
        print("Is black and white: {}".format(detect_color_results_local.color.is_bw_img))
        print("Accent color: {}".format(detect_color_results_local.color.accent_color))
        print("Dominant background color: {}".format(detect_color_results_local.color.dominant_color_background))
        print("Dominant foreground color: {}".format(detect_color_results_local.color.dominant_color_foreground))
        print("Dominant colors: {}".format(detect_color_results_local.color.dominant_colors))
        print()

    def get_detect_color(self):
        tmp_str = ""
        # print("===== Detect Color - local =====\n")
        # Open local image
        local_image = open(self.local_image_path, "rb")
        # Select visual feature(s) you want
        local_image_features = ["color"]
        # Call API with local image and features
        detect_color_results_local = self.computervision_client.analyze_image_in_stream(local_image,
                                                                                        local_image_features)

        # Print results of the color scheme detected
        # tmp_str += "Getting color scheme of the local image: \n"
        # tmp_str += "{}, ".format(detect_color_results_local.color.is_bw_img)
        tmp_str += "{}, ".format(detect_color_results_local.color.accent_color)
        tmp_str += "{}, ".format(detect_color_results_local.color.dominant_color_background)
        tmp_str += "{}, ".format(detect_color_results_local.color.dominant_color_foreground)
        # tmp_str += "{}, ".format(detect_color_results_local.color.dominant_colors)
        tmp_str += "\n"
        return tmp_str


    '''
    Detect Image Types - local
    This example detects an image's type (clip art/line drawing).
    '''

    def detect_image_type(self):
        print("===== Detect Image Types - local =====")
        # Open local image
        local_image_type = open(self.local_image_path, "rb")
        # Select visual feature(s) you want
        local_image_features = VisualFeatureTypes.image_type
        # Call API with local image and features
        detect_type_results_local = self.computervision_client.analyze_image_in_stream(local_image_type,
                                                                                       local_image_features)

        # Print type results with degree of accuracy
        print("Type of local image:")
        if detect_type_results_local.image_type.clip_art_type == 0:
            print("Image is not clip art.")
        elif detect_type_results_local.image_type.line_drawing_type == 1:
            print("Image is ambiguously clip art.")
        elif detect_type_results_local.image_type.line_drawing_type == 2:
            print("Image is normal clip art.")
        else:
            print("Image is good clip art.")

        if detect_type_results_local.image_type.line_drawing_type == 0:
            print("Image is not a line drawing.")
        else:
            print("Image is a line drawing")
        print()

    '''
    Detect Objects - local
    This example detects different kinds of objects with bounding boxes in a local image.
    '''

    def detect_objects(self):
        print("===== Detect Objects - local =====")
        # Get local image with different objects in it
        local_image_objects = open(self.local_image_path, "rb")
        # Call API with local image
        detect_objects_results_local = self.computervision_client.detect_objects_in_stream(local_image_objects)

        # Print results of detection with bounding boxes
        print("Detecting objects in local image:")
        if len(detect_objects_results_local.objects) == 0:
            print("No objects detected.")
        else:
            for object in detect_objects_results_local.objects:
                print("object at location {}, {}, {}, {}".format( \
                    object.rectangle.x, object.rectangle.x + object.rectangle.w, \
                    object.rectangle.y, object.rectangle.y + object.rectangle.h))
        print()

    '''
    Detect Brands - local
    This example detects common brands like logos and puts a bounding box around them.
    '''

    def detect_brands(self):
        print("===== Detect Brands - local =====")
        # Open image file
        local_image_shirt = open(self.local_image_path, "rb")
        # Select the visual feature(s) you want
        local_image_features = ["brands"]
        # Call API with image and features
        detect_brands_results_local = self.computervision_client.analyze_image_in_stream(local_image_shirt,
                                                                                         local_image_features)

        # Print detection results with bounding box and confidence score
        print("Detecting brands in local image: ")
        if len(detect_brands_results_local.brands) == 0:
            print("No brands detected.")
        else:
            for brand in detect_brands_results_local.brands:
                print("'{}' brand detected with confidence {:.1f}% at location {}, {}, {}, {}".format( \
                    brand.name, brand.confidence * 100, brand.rectangle.x, brand.rectangle.x + brand.rectangle.w, \
                    brand.rectangle.y, brand.rectangle.y + brand.rectangle.h))
        print()

    '''
    Generate Thumbnail
    This example creates a thumbnail from both a local and URL image.
    '''

    def generate_thumbnail(self):
        print("===== Generate Thumbnail =====")
        # Generate a thumbnail from a local image
        local_image_thumb = open(self.local_image_path, "rb")

        print("Generating thumbnail from a local image...")
        # Call the API with a local image, set the width/height if desired (pixels)
        # Returns a Generator object, a thumbnail image binary (list).
        thumb_local = self.computervision_client.generate_thumbnail_in_stream(100, 100, local_image_thumb, True)

        # Write the image binary to file
        with open("thumb_local.png", "wb") as f:
            for chunk in thumb_local:
                f.write(chunk)

        # Uncomment/use this if you are writing many images as thumbnails from a list
        # for i, image in enumerate(thumb_local, start=0):
        #      with open('thumb_{0}.jpg'.format(i), 'wb') as f:
        #         f.write(image)

        print("Thumbnail saved to local folder.")

        # Uncomment/use this if you are writing many images as thumbnails from a list
        # for i, image in enumerate(thumb_remote, start=0):
        #      with open('thumb_{0}.jpg'.format(i), 'wb') as f:
        #         f.write(image)

        print()
