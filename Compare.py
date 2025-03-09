import os

# Set the directory where the files are located
directory = "./result"  # Replace with your actual directory path

# Loop over all files in the directory
for filename in os.listdir(directory):
    # Check if the filename starts with "VAE_CMOBO"
    if filename.startswith("VAE_CMOBO"):
        # Construct the new filename by replacing the first occurrence of "VAE_CMOBO" with "VAE-CMOBO"
        new_filename = filename.replace("VAE_CMOBO", "VAE-CMOBO", 1)

        # Build the full file paths
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed '{filename}' to '{new_filename}'")
