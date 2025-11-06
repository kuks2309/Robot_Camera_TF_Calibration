#!/usr/bin/env python3
"""
Test multiple chessboard patterns to find which one works
"""

import cv2
import numpy as np
import os
import glob
import yaml


def test_pattern(image, pattern_size):
    """Test a specific chessboard pattern"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    return ret, corners


def main():
    print("\n" + "="*70)
    print("Test Multiple Chessboard Patterns")
    print("="*70)

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    images_dir = config['storage']['images_dir']

    # Find images
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_files:
        print(f"\n❌ No images found in {images_dir}")
        return 1

    print(f"\n✅ Found {len(image_files)} images")

    # Test image (use first one)
    test_img_path = image_files[0]
    print(f"\nTesting with: {os.path.basename(test_img_path)}")

    image = cv2.imread(test_img_path)
    if image is None:
        print(f"❌ Failed to load image")
        return 1

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Patterns to try (common chessboard sizes)
    patterns_to_try = [
        (9, 6),   # 10x7 squares - from old file (WORKED BEFORE!)
        (6, 9),   # 7x10 squares - rotated
        (10, 7),  # 11x8 squares - from README
        (7, 10),  # 8x11 squares - rotated
        (8, 11),  # From current config
        (11, 8),  # From original config
        (8, 5),   # 9x6 squares
        (5, 8),   # 6x9 squares
    ]

    print(f"\nTrying {len(patterns_to_try)} different patterns...\n")

    results = []

    for pattern in patterns_to_try:
        print(f"Testing {pattern[0]}x{pattern[1]} internal corners...", end=" ")
        ret, corners = test_pattern(image, pattern)

        if ret:
            print(f"✅ DETECTED!")
            results.append(pattern)
        else:
            print(f"❌")

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    if results:
        print(f"\n✅ Found {len(results)} working pattern(s):")
        for pattern in results:
            squares = (pattern[0] + 1, pattern[1] + 1)
            print(f"\n  Pattern: {pattern[0]}x{pattern[1]} internal corners")
            print(f"  Squares: {squares[0]}x{squares[1]}")
            print(f"  Use in config.yaml: pattern_size: [{pattern[0]}, {pattern[1]}]")

        # Test on more images with best pattern
        best_pattern = results[0]
        print(f"\n" + "="*70)
        print(f"Testing pattern {best_pattern[0]}x{best_pattern[1]} on all images...")
        print("="*70 + "\n")

        detected_count = 0
        for i, img_path in enumerate(image_files):
            img_name = os.path.basename(img_path)
            img = cv2.imread(img_path)
            if img is None:
                print(f"[{i+1}/{len(image_files)}] {img_name}: ❌ Load failed")
                continue

            ret, _ = test_pattern(img, best_pattern)
            status = "✅" if ret else "❌"
            print(f"[{i+1}/{len(image_files)}] {img_name}: {status}")
            if ret:
                detected_count += 1

        print(f"\nDetection rate: {detected_count}/{len(image_files)} ({100*detected_count/len(image_files):.1f}%)")

        if detected_count >= 10:
            print(f"✅ GOOD! Enough images for calibration")
        else:
            print(f"⚠️  Only {detected_count} images detected (need ≥10)")

    else:
        print("\n❌ No patterns detected!")
        print("\nPossible issues:")
        print("  1. Chessboard not fully visible in image")
        print("  2. Chessboard has different size than tested")
        print("  3. Image quality/lighting issues")
        print("  4. Wrong chessboard type (need standard chessboard, not ChArUco)")

    print("="*70 + "\n")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
