import gdown
import shutil
import zipfile

# File ID from the Google Drive link
file_id = '1mjYo4ew0pTVD3tJhwh2EGCiLENmG0RRo'

# URL template to download from Google Drive
url = f'https://drive.google.com/uc?id={file_id}'

# Output file name
output = 'downloaded_file.zip'

# Download the file
gdown.download(url, output, quiet=False)

def unzip_folder(zip_name, extract_dir):
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print(f"{zip_name} has been extracted to {extract_dir}")

zip_to_extract = 'saved_models.zip'  # Replace 'saved_models.zip' with your zip file name
extract_to_folder = 'saved_models'  # Replace 'extracted_models' with the desired extraction folder name

if os.path.exists(zip_to_extract) and zipfile.is_zipfile(zip_to_extract):
    unzip_folder(zip_to_extract, extract_to_folder)
else:
    print(f"File '{zip_to_extract}' does not exist or is not a valid ZIP file.")

file_to_delete = 'saved_models.zip'  # Replace with the file you want to delete

if os.path.exists(file_to_delete):
    os.remove(file_to_delete)
    print(f"{file_to_delete} has been deleted.")
else:
    print(f"File '{file_to_delete}' does not exist.")