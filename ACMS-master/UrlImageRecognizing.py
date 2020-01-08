class UrlCV:
    def __init__(self, computervision_client, url):
        self.remote_image_url = url
        self.computervision_client = computervision_client

    def get_image_url(self):
        return self.remote_image_url

    def set_image_url(self, url):
        self.remote_image_url = url

    def set_computervision_client(self, computervision_client):
        self.computervision_client = computervision_client

    def describe_image(self):
        '''
        Describe an Image - remote
        This example describes the contents of an image with the confidence score.
        '''
        print("===== Describe an image - remote =====")
        # Call API
        description_results = self.computervision_client.describe_image(self.remote_image_url)

        # Get the captions (descriptions) from the response, with confidence level
        print("Description of remote image: ")
        if (len(description_results.captions) == 0):
            print("No description detected.")
        else:
            for caption in description_results.captions:
                print("'{}' with confidence {:.2f}%".format(caption.text, caption.confidence * 100))

    def categorize_image(self):
        '''
        Categorize an Image - remote
        This example extracts (general) categories from a remote image with a confidence score.
        '''
        print("===== Categorize an image - remote =====")
        # Select the visual feature(s) you want.
        remote_image_features = ["categories"]
        # Call API with URL and features
        categorize_results_remote = self.computervision_client.analyze_image(self.remote_image_url, remote_image_features)

        # Print results with confidence score
        print("Categories from remote image: ")
        if (len(categorize_results_remote.categories) == 0):
            print("No categories detected.")
        else:
            for category in categorize_results_remote.categories:
                print("'{}' with confidence {:.2f}%".format(category.name, category.score * 100))

    def tag_image(self):
        '''
        Tag an Image - remote
        This example returns a tag (key word) for each thing in the image.
        '''
        print("===== Tag an image - remote =====")
        # Call API with remote image
        tags_result_remote = self.computervision_client.tag_image(self.remote_image_url)

        # Print results with confidence score
        print("Tags in the remote image: ")
        if (len(tags_result_remote.tags) == 0):
            print("No tags detected.")
        else:
            for tag in tags_result_remote.tags:
                print("'{}' with confidence {:.2f}%".format(tag.name, tag.confidence * 100))

    def color_image(self):
        '''
        Detect Color - remote
        This example detects the different aspects of its color scheme in a remote image.
        '''
        print("===== Detect Color - remote =====")
        # Select the feature(s) you want
        remote_image_features = ["color"]
        # Call API with URL and features
        detect_color_results_remote = self.computervision_client.analyze_image(self.remote_image_url, remote_image_features)

        # Print results of color scheme
        print("Getting color scheme of the remote image: ")
        print("Is black and white: {}".format(detect_color_results_remote.color.is_bw_img))
        print("Accent color: {}".format(detect_color_results_remote.color.accent_color))
        print("Dominant background color: {}".format(detect_color_results_remote.color.dominant_color_background))
        print("Dominant foreground color: {}".format(detect_color_results_remote.color.dominant_color_foreground))
        print("Dominant colors: {}".format(detect_color_results_remote.color.dominant_colors))
