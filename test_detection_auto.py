#!/usr/bin/env python3
"""
Automatic chessboard detection test - shows images automatically
"""

import cv2
import numpy as np
import os
import glob
import yaml
import time


def detect_chessboard(image, pattern_size, square_size_mm):
    """Detect chessboard in image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # Create annotated image
    annotated = image.copy()

    if ret:
        # Refine corner detection
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw detected corners
        cv2.drawChessboardCorners(annotated, pattern_size, corners_refined, ret)

        # Add success text
        cv2.putText(annotated, f"DETECTED: {pattern_size[0]}x{pattern_size[1]} corners",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        cv2.putText(annotated, f"Square size: {square_size_mm}mm",
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return True, corners_refined, annotated
    else:
        # Add failure text
        cv2.putText(annotated, f"NOT DETECTED: {pattern_size[0]}x{pattern_size[1]}",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        return False, None, annotated


def main():
    print("\n" + "="*70)
    print("Automatic Chessboard Detection Test")
    print("="*70)

    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    pattern_size = tuple(config['chessboard']['pattern_size'])
    square_size_mm = config['chessboard']['square_size_mm']
    images_dir = config['storage']['images_dir']

    print(f"\nConfiguration:")
    print(f"  Pattern: {pattern_size[0]}x{pattern_size[1]} internal corners")
    print(f"  Square size: {square_size_mm}mm")
    print(f"  Images directory: {images_dir}")

    # Find all images
    image_files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

    if not image_files:
        print(f"\n❌ No images found in {images_dir}")
        return 1

    print(f"\n✅ Found {len(image_files)} images")
    print("\nAuto-playing images (1 second each)")
    print("Press 'q' in OpenCV window to quit\n")

    # Statistics
    detected_count = 0
    failed_images = []

    window_name = "Chessboard Detection Test (Auto)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    for i, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f"[{i+1}/{len(image_files)}] {img_name}...", end=" ")

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print("❌ Failed to load")
            failed_images.append(img_name)
            continue

        # Detect chessboard
        success, corners, annotated = detect_chessboard(image, pattern_size, square_size_mm)

        if success:
            print("✅")
            detected_count += 1
        else:
            print("❌")
            failed_images.append(img_name)

        # Add image info overlay
        h, w = annotated.shape[:2]
        cv2.putText(annotated, f"Image {i+1}/{len(image_files)}: {img_name}",
                   (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated, f"Detection rate: {detected_count}/{i+1} ({100*detected_count/(i+1):.1f}%)",
                   (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(annotated, "Auto-playing (press 'q' to quit)",
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Resize for display
        display = cv2.resize(annotated, (1280, 720))
        cv2.imshow(window_name, display)

        # Wait 1 second or until 'q' is pressed
        key = cv2.waitKey(1000) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("\n\nUser quit")
            break

    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "="*70)
    print("Detection Results")
    print("="*70)
    print(f"Total images: {len(image_files)}")
    print(f"Successfully detected: {detected_count} ({100*detected_count/len(image_files):.1f}%)")
    print(f"Failed to detect: {len(failed_images)} ({100*len(failed_images)/len(image_files):.1f}%)")

    if failed_images:
        print(f"\nFailed images ({len(failed_images)}):")
        for img_name in failed_images[:10]:
            print(f"  - {img_name}")
        if len(failed_images) > 10:
            print(f"  ... and {len(failed_images) - 10} more")

    print("="*70)

    if detected_count >= 10:
        print("\n✅ GOOD! You have enough images for calibration (≥10)")
        print("\nNext step: Run compute_calibration.py")
    else:
        print(f"\n⚠️  Warning: Only {detected_count} images detected.")
        print(f"   Recommended: ≥10 images for good calibration")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        cv2.destroyAllWindows()
        exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
