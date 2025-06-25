

import numpy as np
from PIL import Image
import random
import os


class ImageEncryption:
    """
    A simple image encryption tool using pixel manipulation techniques.
    """

    def __init__(self):
        self.seed_value = None

    def load_image(self, image_path):
        """
        Load an image and convert it to numpy array.

        Args:
            image_path (str): Path to the image file

        Returns:
            numpy.ndarray: Image as numpy array
        """
        try:
            img = Image.open(image_path)
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    def save_image(self, image_array, output_path):
        """
        Save numpy array as image file.

        Args:
            image_array (numpy.ndarray): Image data as numpy array
            output_path (str): Path to save the image
        """
        try:
            # Ensure values are in valid range
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
            img = Image.fromarray(image_array)
            img.save(output_path)
            print(f"Image saved to: {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")

    def xor_encrypt_decrypt(self, image_array, key):
        """
        Encrypt/decrypt image using XOR operation with a key.

        Args:
            image_array (numpy.ndarray): Image data
            key (int): Encryption key (0-255)

        Returns:
            numpy.ndarray: Processed image
        """
        return image_array ^ key

    def pixel_swap_encrypt(self, image_array, seed):
        """
        Encrypt image by swapping pixels based on a seed.

        Args:
            image_array (numpy.ndarray): Image data
            seed (int): Random seed for reproducible swapping

        Returns:
            numpy.ndarray: Encrypted image
        """
        self.seed_value = seed
        encrypted = image_array.copy()
        height, width, channels = encrypted.shape

        # Create list of all pixel positions
        positions = [(i, j) for i in range(height) for j in range(width)]

        # Set seed for reproducible randomness
        random.seed(seed)

        # Shuffle positions
        shuffled_positions = positions.copy()
        random.shuffle(shuffled_positions)

        # Create new image with swapped pixels
        result = np.zeros_like(encrypted)
        for original_pos, new_pos in zip(positions, shuffled_positions):
            result[new_pos[0], new_pos[1]] = encrypted[original_pos[0], original_pos[1]]

        return result

    def pixel_swap_decrypt(self, encrypted_array, seed):
        """
        Decrypt image by reversing pixel swapping.

        Args:
            encrypted_array (numpy.ndarray): Encrypted image data
            seed (int): Same seed used for encryption

        Returns:
            numpy.ndarray: Decrypted image
        """
        height, width, channels = encrypted_array.shape

        # Create list of all pixel positions
        positions = [(i, j) for i in range(height) for j in range(width)]

        # Set seed for reproducible randomness
        random.seed(seed)

        # Shuffle positions (same as encryption)
        shuffled_positions = positions.copy()
        random.shuffle(shuffled_positions)

        # Reverse the swapping
        result = np.zeros_like(encrypted_array)
        for original_pos, new_pos in zip(positions, shuffled_positions):
            result[original_pos[0], original_pos[1]] = encrypted_array[new_pos[0], new_pos[1]]

        return result

    def mathematical_encrypt(self, image_array, key):
        """
        Encrypt image using mathematical operations.

        Args:
            image_array (numpy.ndarray): Image data
            key (int): Mathematical key

        Returns:
            numpy.ndarray: Encrypted image
        """
        # Apply mathematical transformation: (pixel + key) mod 256
        encrypted = (image_array.astype(np.int16) + key) % 256
        return encrypted.astype(np.uint8)

    def mathematical_decrypt(self, encrypted_array, key):
        """
        Decrypt image using reverse mathematical operations.

        Args:
            encrypted_array (numpy.ndarray): Encrypted image data
            key (int): Same key used for encryption

        Returns:
            numpy.ndarray: Decrypted image
        """
        # Reverse transformation: (pixel - key) mod 256
        decrypted = (encrypted_array.astype(np.int16) - key) % 256
        return decrypted.astype(np.uint8)

    def rgb_channel_shift(self, image_array, shift_pattern):
        """
        Encrypt by shifting RGB channels.

        Args:
            image_array (numpy.ndarray): Image data
            shift_pattern (str): Pattern like "RGB" -> "GBR"

        Returns:
            numpy.ndarray: Image with shifted channels
        """
        if shift_pattern == "GBR":
            # R->G, G->B, B->R
            return image_array[:, :, [1, 2, 0]]
        elif shift_pattern == "BRG":
            # R->B, G->R, B->G
            return image_array[:, :, [2, 0, 1]]
        else:
            return image_array


def get_valid_key(prompt, min_val=0, max_val=255):
    """Get valid key input from user."""
    while True:
        try:
            key = int(input(prompt))
            if min_val <= key <= max_val:
                return key
            else:
                print(f"Please enter a number between {min_val} and {max_val}.")
        except ValueError:
            print("Please enter a valid integer.")


def main():
    """Main program function."""
    encryptor = ImageEncryption()

    print("=== Simple Image Encryption Tool ===")
    print("Supported formats: PNG, JPG, JPEG, BMP")
    print()

    while True:
        print("\nEncryption Methods:")
        print("1. XOR Encryption/Decryption")
        print("2. Pixel Swapping")
        print("3. Mathematical Transformation")
        print("4. RGB Channel Shifting")
        print("5. Exit")

        choice = input("\nChoose encryption method (1-5): ").strip()

        if choice == '5':
            print("Thanks for using the Image Encryption Tool!")
            break

        if choice not in ['1', '2', '3', '4']:
            print("Invalid choice. Please try again.")
            continue

        # Get operation type
        print("\nOperation:")
        print("1. Encrypt")
        print("2. Decrypt")
        operation = input("Choose operation (1-2): ").strip()

        if operation not in ['1', '2']:
            print("Invalid operation choice.")
            continue

        # Get input image path
        input_path = input("Enter input image path: ").strip()
        if not os.path.exists(input_path):
            print("Image file not found!")
            continue

        # Load image
        image_data = encryptor.load_image(input_path)
        if image_data is None:
            continue

        # Get output path
        output_path = input("Enter output image path: ").strip()

        # Process based on chosen method
        if choice == '1':  # XOR
            key = get_valid_key("Enter XOR key (0-255): ")
            result = encryptor.xor_encrypt_decrypt(image_data, key)

        elif choice == '2':  # Pixel Swapping
            seed = get_valid_key("Enter seed value (any integer): ", min_val=0, max_val=999999)
            if operation == '1':  # Encrypt
                result = encryptor.pixel_swap_encrypt(image_data, seed)
            else:  # Decrypt
                result = encryptor.pixel_swap_decrypt(image_data, seed)

        elif choice == '3':  # Mathematical
            key = get_valid_key("Enter mathematical key (0-255): ")
            if operation == '1':  # Encrypt
                result = encryptor.mathematical_encrypt(image_data, key)
            else:  # Decrypt
                result = encryptor.mathematical_decrypt(image_data, key)

        elif choice == '4':  # RGB Channel Shifting
            if operation == '1':  # Encrypt
                print("Channel shift patterns:")
                print("1. RGB -> GBR")
                print("2. RGB -> BRG")
                pattern_choice = input("Choose pattern (1-2): ").strip()
                pattern = "GBR" if pattern_choice == '1' else "BRG"
                result = encryptor.rgb_channel_shift(image_data, pattern)
            else:  # Decrypt
                print("Reverse channel shift patterns:")
                print("1. GBR -> RGB")
                print("2. BRG -> RGB")
                pattern_choice = input("Choose reverse pattern (1-2): ").strip()
                # For decryption, we need to reverse the shift
                if pattern_choice == '1':  # Reverse GBR
                    result = encryptor.rgb_channel_shift(image_data, "BRG")
                else:  # Reverse BRG
                    result = encryptor.rgb_channel_shift(image_data, "GBR")

        # Save result
        encryptor.save_image(result, output_path)
        print(f"Operation completed successfully!")


def demo():
    """Demonstration of the encryption tool."""
    print("=== Demo Mode ===")
    print("This demo shows how the different encryption methods work.")
    print("For a full demo, you would need actual image files.")
    print()

    # Create a simple test pattern
    test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    encryptor = ImageEncryption()

    print("1. XOR Encryption Demo:")
    encrypted_xor = encryptor.xor_encrypt_decrypt(test_image, 123)
    decrypted_xor = encryptor.xor_encrypt_decrypt(encrypted_xor, 123)
    print(f"   Original matches decrypted: {np.array_equal(test_image, decrypted_xor)}")

    print("2. Mathematical Encryption Demo:")
    encrypted_math = encryptor.mathematical_encrypt(test_image, 50)
    decrypted_math = encryptor.mathematical_decrypt(encrypted_math, 50)
    print(f"   Original matches decrypted: {np.array_equal(test_image, decrypted_math)}")

    print("3. Pixel Swapping Demo:")
    encrypted_swap = encryptor.pixel_swap_encrypt(test_image, 12345)
    decrypted_swap = encryptor.pixel_swap_decrypt(encrypted_swap, 12345)
    print(f"   Original matches decrypted: {np.array_equal(test_image, decrypted_swap)}")

    print("\nDemo completed! All methods successfully encrypt and decrypt.")


if __name__ == "__main__":
    # Run demo
    demo()

    # Run main program
    main()
     
