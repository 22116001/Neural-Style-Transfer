# Neural Style Transfer

Transform ordinary photographs into masterpieces using the artistic styles.

## Overview

Welcome to my Neural Style Transfer project, where art meets technology. This project leverages deep learning techniques to blend the content of one image with the style of another, producing visually appealing and artistically coherent results. The model is designed to balance content preservation with stylistic transformation, offering a tool to create unique and personalized artwork.

## Features

- **Style Transfer**: Extracts stylistic features from an artwork and applies them to a different image while preserving the content structure.
- **Dynamic Loading**: Allows for the dynamic upload of new content and style images.
- **Optimization**: Utilizes optimization algorithms and perceptual loss functions for effective style transfer.
- **GPU Acceleration**: Employs CUDA for GPU acceleration when available.

## Installation

1. Download the code repository.
2. Open the "Neural_Style_Transfer5.ipynb" file.
3. In cell 2, paste the path of the folder containing the content and style images.
4. In cell 12, specify the filenames of the content and style images to get the desired output.

## Usage

1. **Mount Google Drive** (if using Google Colab):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Adjust Image Directory**:
   ```python
   image_directory = "/content/drive/MyDrive/Neural Style Transfer/Testing images/"
   ```

3. **Set Up GPU/CPU Device**:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

4. **Load and Resize Images**:
   ```python
   content_path = image_directory + "content1.jpg"
   style_path = image_directory + "style1.jpg"
   content_img, style_img = dynamic_load_images(content_path, style_path, imsize)
   ```

5. **Run Style Transfer**:
   ```python
   output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_steps=300)
   ```

6. **Display the Output Image**:
   ```python
   plt.figure()
   show_image(output, title='Output Image')
   plt.show()
   ```

## Examples

Content image
<br>
<img width="500" alt="image" src="https://github.com/22116001/Neural-Style-Transfer/assets/118989888/558a93a9-c0d5-40da-86d4-ad419d2a0948">
<br>
Style image
<br>
<img width="494" alt="image" src="https://github.com/22116001/Neural-Style-Transfer/assets/118989888/208b0c74-e46c-415c-a0d4-e0436c55eb1d">
<br>
Final Generated Image
<br>
<img width="496" alt="image" src="https://github.com/22116001/Neural-Style-Transfer/assets/118989888/2344ac2a-2e12-4241-8fdd-d147da1cfd3f">


## Architecture and Implementation Details

### Components

- **Neural Network**: Built using the pre-trained VGG-19 model from torchvision.
- **Loss Functions**: ContentLoss and StyleLoss are based on MSE loss and Gram Matrix computation.
- **Normalization**: Standardizes input images to match VGG-19's training data statistics.

### Model Training

- **Content and Style Layers**: Configured to extract features from content and style images effectively.

## References

Here are some key papers and resources I referred to during the project:

- https://arxiv.org/pdf/1508.06576.pdf
- https://arxiv.org/pdf/1603.08155.pdf
- https://arxiv.org/pdf/1703.06868.pdf
- https://arxiv.org/pdf/1705.06830.pdf
- https://arxiv.org/pdf/1804.03547.pdf
- https://arxiv.org/pdf/1912.07921.pdf

## Future Enhancements

Looking ahead, I plan to:

- Support additional artistic styles.
- Integrate a web interface for easier image manipulation.
- Optimize performance for faster processing.
