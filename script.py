import os

# Path to the folder containing the images
folder_path = "hagaralttai"

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"Folder not found: {folder_path}")
else:
    # Loop through the files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is a .jpg file
        if filename.endswith(".jpg"):
            # Extract the numeric part before "_1" and ensure it's numeric
            file_base = filename.split("_")[0]
            if file_base.isdigit():
                file_number = int(file_base)  # Convert the numeric part to an integer
                # If the number is between 10000 and 19378, delete the file
                if 10000 <= file_number <= 19378:
                    file_path = os.path.join(folder_path, filename)
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}: {e}")

    print("Deletion process completed.")
