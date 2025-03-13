import os
import imghdr

def verify_dataset(directory):
    """
    Verify dataset integrity
    - Check directory structure
    - Validate image files
    """
    print(f"Verifying dataset in {directory}")
    
    # Track valid and invalid images
    valid_images = 0
    invalid_images = 0
    invalid_files = []
    class_image_counts = {}
    
    # Iterate through classes
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_path):
            continue
        
        # Count images in this class
        class_images = 0
        
        # Check images in this class
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            
            try:
                # Verify it's an image
                img_type = imghdr.what(img_path)
                if img_type in ['jpeg', 'png', 'gif', 'bmp', 'jpg']:
                    valid_images += 1
                    class_images += 1
                else:
                    invalid_images += 1
                    invalid_files.append(img_path)
            except Exception as e:
                invalid_images += 1
                invalid_files.append(img_path)
        
        class_image_counts[class_name] = class_images
    
    # Print summary
    print("\nDataset Verification Summary:")
    print(f"Total Valid Images: {valid_images}")
    print(f"Total Invalid Images: {invalid_images}")
    print("\nImages per Class:")
    for class_name, count in class_image_counts.items():
        print(f"{class_name}: {count} images")
    
    if invalid_files:
        print("\nInvalid Files:")
        for file in invalid_files:
            print(file)

# Verify train and test datasets
print("Verifying Training Dataset:")
verify_dataset('data/train')
print("\nVerifying Test Dataset:")
verify_dataset('data/test')