import unittest
import os
import sys
import shutil
from pathlib import Path

# --- PATH SETUP ---
# 1. Get the folder where this test file is located (e.g., .../project/tests)
TEST_DIR = Path(__file__).resolve().parent

# 2. Get the project root (e.g., .../project)
PROJECT_ROOT = TEST_DIR.parent

# 3. Define the source directory (e.g., .../project/src)
SRC_DIR = PROJECT_ROOT / "src"

# 4. Add src to Python's search path so we can import 'podcast_backend'
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# --- IMPORT MODULE ---
try:
    import podcast_backend
except ImportError:
    # Fallback: Try looking in the current directory if running from src
    try:
        import podcast_backend
    except ImportError:
        print(f"❌ CRITICAL ERROR: Could not find 'podcast_backend.py'.")
        print(f"   - Looked in: {SRC_DIR}")
        print("   - Make sure your folder structure is correct:")
        print("     project/")
        print("       ├── src/podcast_backend.py")
        print("       └── tests/test_podcast_backend.py")
        sys.exit(1)

class TestPodcastBackend(unittest.TestCase):

    def test_format_time(self):
        """Test if seconds are correctly converted to MM:SS format."""
        print("\nTesting Time Formatting...")
        self.assertEqual(podcast_backend.format_time(60), "01:00")
        self.assertEqual(podcast_backend.format_time(65), "01:05")
        self.assertEqual(podcast_backend.format_time(0), "00:00")
        self.assertEqual(podcast_backend.format_time(90.5), "01:30")
        print("✅ Time Formatting OK")

    def test_extract_keywords_simple(self):
        """Test if keyword extraction works on a simple string."""
        print("\nTesting Keyword Extraction...")
        dummy_text = "apple banana apple orange banana apple grape"
        
        # We expect a string back
        result = podcast_backend.extract_keywords_text(dummy_text, top_n=2)
        
        # Check if result is a string and contains expected words
        self.assertIsInstance(result, str)
        self.assertIn("apple", result)
        print("✅ Keyword Extraction OK")

    def test_setup_directories(self):
        """Test if the directory creation function actually makes folders."""
        print("\nTesting Directory Setup...")
        
        # Use a temporary test path inside tests folder to avoid messing up real data
        test_base_dir = TEST_DIR / "temp_test_data"
        
        # Run the function
        dirs = podcast_backend.setup_directories(str(test_base_dir))
        
        # Check if folders exist
        self.assertTrue(os.path.exists(dirs["audio"]))
        self.assertTrue(os.path.exists(dirs["transcripts"]))
        
        # Cleanup: Remove the test folder after testing
        if os.path.exists(test_base_dir):
            shutil.rmtree(test_base_dir)
        print("✅ Directory Setup OK")

if __name__ == '__main__':
    unittest.main()