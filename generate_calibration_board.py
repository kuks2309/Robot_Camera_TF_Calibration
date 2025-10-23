#!/usr/bin/env python3
"""
Generate calibration boards for printing.

This script generates either ChArUco or Chessboard patterns
that can be printed for hand-eye calibration.
"""

import cv2
import numpy as np
import argparse
import sys


def create_charuco_board(grid_width=8, grid_height=6,
                        square_size=352, marker_size=247,
                        dpi=300, output_file="charuco_board.png"):
    """
    Create ChArUco calibration board.

    Args:
        grid_width: Number of squares in width
        grid_height: Number of squares in height
        square_size: Square size in pixels
        marker_size: ArUco marker size in pixels
        dpi: DPI for printing
        output_file: Output image filename
    """
    # Create ArUco dictionary and ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    board = cv2.aruco.CharucoBoard(
        (grid_width, grid_height),
        square_size / dpi * 0.0254,  # Convert pixels to meters
        marker_size / dpi * 0.0254,
        aruco_dict
    )

    # Generate board image
    img_width = grid_width * square_size
    img_height = grid_height * square_size
    img = board.generateImage((img_width, img_height), marginSize=0, borderBits=1)

    # Save image
    cv2.imwrite(output_file, img)

    # Calculate physical dimensions
    width_inches = img_width / dpi
    height_inches = img_height / dpi
    square_mm = square_size / dpi * 25.4
    marker_mm = marker_size / dpi * 25.4

    print(f"\n✅ ChArUco board generated: {output_file}")
    print(f"   Grid: {grid_width}x{grid_height}")
    print(f"   Image size: {img_width}x{img_height} pixels")
    print(f"   Physical size: {width_inches:.2f}\" x {height_inches:.2f}\" ({width_inches*25.4:.1f}mm x {height_inches*25.4:.1f}mm)")
    print(f"   Square size: {square_mm:.2f}mm")
    print(f"   Marker size: {marker_mm:.2f}mm")
    print(f"   DPI: {dpi}")
    print(f"\n📋 Print settings:")
    print(f"   - Print at {dpi} DPI")
    print(f"   - Disable scaling/fit-to-page")
    print(f"   - Use high-quality paper")
    print(f"   - Mount on rigid board")


def create_chessboard(pattern_width=10, pattern_height=7,
                     square_size=352, dpi=300,
                     output_file="chessboard.png"):
    """
    Create chessboard calibration pattern.

    Args:
        pattern_width: Number of squares in width
        pattern_height: Number of squares in height
        square_size: Square size in pixels
        dpi: DPI for printing
        output_file: Output image filename

    Note: pattern_width x pattern_height gives (pattern_width-1) x (pattern_height-1) internal corners
    """
    img_width = pattern_width * square_size
    img_height = pattern_height * square_size

    # Create chessboard pattern
    img = np.zeros((img_height, img_width), dtype=np.uint8)

    for i in range(pattern_height):
        for j in range(pattern_width):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                img[y1:y2, x1:x2] = 255

    # Save image
    cv2.imwrite(output_file, img)

    # Calculate physical dimensions
    width_inches = img_width / dpi
    height_inches = img_height / dpi
    square_mm = square_size / dpi * 25.4

    print(f"\n✅ Chessboard generated: {output_file}")
    print(f"   Grid: {pattern_width}x{pattern_height} squares")
    print(f"   Internal corners: {pattern_width-1}x{pattern_height-1}")
    print(f"   Image size: {img_width}x{img_height} pixels")
    print(f"   Physical size: {width_inches:.2f}\" x {height_inches:.2f}\" ({width_inches*25.4:.1f}mm x {height_inches*25.4:.1f}mm)")
    print(f"   Square size: {square_mm:.2f}mm")
    print(f"   DPI: {dpi}")
    print(f"\n📋 Print settings:")
    print(f"   - Print at {dpi} DPI")
    print(f"   - Disable scaling/fit-to-page")
    print(f"   - Use high-quality paper")
    print(f"   - Mount on rigid board")


def main():
    parser = argparse.ArgumentParser(
        description='Generate calibration boards for hand-eye calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default ChArUco board (8x6, 30mm squares)
  python3 generate_calibration_board.py --type charuco

  # Generate default Chessboard (10x7 squares, 9x6 corners, 25mm squares)
  python3 generate_calibration_board.py --type chessboard

  # Custom ChArUco board
  python3 generate_calibration_board.py --type charuco --grid 10 8 --square 40

  # Custom Chessboard
  python3 generate_calibration_board.py --type chessboard --grid 12 9 --square 30

Board Specifications:
  ChArUco (default):
    - Grid: 8x6 (8 columns x 6 rows)
    - Square size: 30mm
    - Marker size: 21mm (70% of square)
    - Physical size: ~240mm x 180mm

  Chessboard (default):
    - Grid: 10x7 squares (9x6 internal corners)
    - Square size: 25mm
    - Physical size: ~250mm x 175mm

Print Instructions:
  1. Print at 300 DPI without scaling
  2. Measure printed squares to verify size
  3. Mount on rigid flat surface (foam board, acrylic)
  4. Ensure board is completely flat for accurate calibration
        """
    )

    parser.add_argument('--type', choices=['charuco', 'chessboard'],
                       default='charuco',
                       help='Type of calibration board (default: charuco)')

    parser.add_argument('--grid', type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'),
                       help='Grid size: WIDTH HEIGHT (ChArUco: squares, Chessboard: squares)')

    parser.add_argument('--square', type=float,
                       help='Square size in millimeters')

    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for printing (default: 300)')

    parser.add_argument('--output', type=str,
                       help='Output filename (default: auto-generated)')

    args = parser.parse_args()

    # Set defaults based on board type
    if args.type == 'charuco':
        default_grid = (8, 6)
        default_square_mm = 30.0
        default_output = 'charuco_board.png'
    else:  # chessboard
        default_grid = (10, 7)
        default_square_mm = 25.0
        default_output = 'chessboard.png'

    # Override with user arguments
    grid = tuple(args.grid) if args.grid else default_grid
    square_mm = args.square if args.square else default_square_mm
    output_file = args.output if args.output else default_output

    # Convert mm to pixels at given DPI
    square_pixels = int(square_mm / 25.4 * args.dpi)

    print("="*80)
    print(f"Generating {args.type.upper()} Calibration Board")
    print("="*80)

    if args.type == 'charuco':
        marker_pixels = int(square_pixels * 0.7)  # Marker is 70% of square
        create_charuco_board(
            grid_width=grid[0],
            grid_height=grid[1],
            square_size=square_pixels,
            marker_size=marker_pixels,
            dpi=args.dpi,
            output_file=output_file
        )
    else:  # chessboard
        create_chessboard(
            pattern_width=grid[0],
            pattern_height=grid[1],
            square_size=square_pixels,
            dpi=args.dpi,
            output_file=output_file
        )

    print("\n" + "="*80)
    print("✅ Board generation complete!")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
