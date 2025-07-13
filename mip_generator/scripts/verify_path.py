import os
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_path_access():
    """
    Verifies read/write access to the directory specified by the HF_HOME environment variable.
    """
    logging.info("--- Starting Path Access Verification Test ---")

    # 1. Get the target path from the environment variable
    target_path = os.environ.get('HF_HOME')
    if not target_path:
        logging.error("HF_HOME environment variable is not set.")
        logging.error("Please set it to your desired cache directory before running the script.")
        logging.error("Example: export HF_HOME=/path/to/your/cache")
        return

    logging.info(f"Target directory (from HF_HOME): {target_path}")

    # 2. Ensure the directory exists
    try:
        os.makedirs(target_path, exist_ok=True)
        logging.info(f"Directory exists or was created successfully.")
    except Exception as e:
        logging.error(f"Failed to create directory at {target_path}: {e}")
        return

    # 3. Define the test file path and content
    test_file_path = os.path.join(target_path, "access_test.txt")
    test_content = "Hello from the MIP Generator! Access successful."
    logging.info(f"Test file will be created at: {test_file_path}")

    # 4. Write the test file
    try:
        with open(test_file_path, 'w') as f:
            f.write(test_content)
        logging.info("Successfully wrote test file.")
    except Exception as e:
        logging.error(f"Failed to write to {test_file_path}: {e}")
        return

    # 5. Read the test file back
    try:
        with open(test_file_path, 'r') as f:
            retrieved_content = f.read()
        logging.info("Successfully read test file.")
        
        # 6. Verify the content
        if retrieved_content == test_content:
            logging.info("SUCCESS: Content matches. Read/write access verified.")
        else:
            logging.error("FAILURE: Content mismatch. Something went wrong.")
            logging.error(f"  - Expected: '{test_content}'")
            logging.error(f"  - Retrieved: '{retrieved_content}'")

    except Exception as e:
        logging.error(f"Failed to read from {test_file_path}: {e}")
        return
    
    finally:
        # 7. Clean up the test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
            logging.info(f"Cleaned up test file: {test_file_path}")

    logging.info("--- Path Access Verification Test Complete ---")


if __name__ == "__main__":
    verify_path_access()
