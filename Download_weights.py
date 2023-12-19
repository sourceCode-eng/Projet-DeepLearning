import gdown
import shutil
import zipfile

# File ID from the Google Drive link
file_id = '1HS-FrsrsCI_RzUnZJdIerjBJHiFBPxuI'

# URL template to download from Google Drive
url = f'https://drive.google.com/uc?id={file_id}'

# Output file name
output = 'downloaded_file.zip'

# Download the file
gdown.download(url, output, quiet=False)

# Unzip the downloaded file (if it's a zip file)
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('extracted_folder')  # Replace 'extracted_folder' with the desired extraction path

# Clean up - remove the downloaded zip file
shutil.rmtree(output)