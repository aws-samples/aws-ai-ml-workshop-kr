#!/usr/bin/env python3
"""
Test function for process_image_with_llm function with real images
"""

import os
import sys

# Add parent directory to path to import the module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from processing_local_img import process_image_with_llm


class TestProcessImageWithLLM:
    """Test cases for process_image_with_llm function with real images"""
    
    def __init__(self):
        self.figures_path = os.path.join(os.path.dirname(__file__), '..', 'figures')
        self.test_images = [
            'figure-2-1.jpg',
            'figure-3-2.jpg'
        ]
    
    def test_real_images(self):
        """Test with actual images from figures directory"""
        print("🧪 Testing process_image_with_llm with real images and LLM calls\n")
        
        test_domain = "비즈니스"
        test_num_img_questions = "2"
        
        for image_name in self.test_images:
            image_path = os.path.join(self.figures_path, image_name)
            
            if not os.path.exists(image_path):
                print(f"❌ Image not found: {image_path}")
                continue
                
            print(f"📸 Testing image: {image_name}")
            print(f"   Path: {image_path}")
            print(f"   Domain: {test_domain}")
            print(f"   Questions: {test_num_img_questions}")
            
            try:
                # Call the actual function with real LLM
                result = process_image_with_llm(image_path, test_domain, test_num_img_questions)
                
                print(f"✅ Success! Generated {len(result)} Q&A pairs")
                
                # Print the results
                for i, qa in enumerate(result, 1):
                    print(f"\n   Q{i}: {qa.get('QUESTION', 'N/A')}")
                    print(f"   A{i}: {qa.get('ANSWER', 'N/A')}")
                    print(f"   Source: {qa.get('source', 'N/A')}")
                    print(f"   Image: {qa.get('image_path', 'N/A')}")
                
            except Exception as e:
                print(f"❌ Error processing {image_name}: {str(e)}")
            
            print("-" * 80)
    
    def run_test(self):
        """Run the real image tests"""
        self.test_real_images()


def main():
    """Main function to run tests"""
    tester = TestProcessImageWithLLM()
    tester.run_test()


if __name__ == "__main__":
    main()